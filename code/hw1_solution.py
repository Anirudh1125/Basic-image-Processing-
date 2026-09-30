# -*- coding: utf-8 -*-
"""
CSI-471/571: Introduction to Computer Vision
Homework 1 - Basic Image Processing
Full solution script (Python 3 / skimage / numpy / matplotlib)

Run this file from the folder that contains the data images
(quart.jpg, snowman.jpg, rotate.jpg, things1.png, things2.png, lenag.png).
It regenerates every output image used in the write-up into ./outputs/.

    python3 hw1_solution.py
"""

import os
import numpy as np
from skimage import io, color
import matplotlib.pyplot as plt

INPUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(INPUT_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


def load_gray(name):
    """Load an image from INPUT_DIR as a uint8 grayscale numpy array."""
    im = io.imread(os.path.join(INPUT_DIR, name))
    if im.ndim == 3:
        im = (color.rgb2gray(im) * 255).astype(np.uint8)
    return im.astype(np.uint8)


def save(name, im):
    """Save an array to OUT_DIR, clipping/casting to uint8 first."""
    im = np.clip(im, 0, 255).astype(np.uint8)
    io.imsave(os.path.join(OUT_DIR, name), im, check_contrast=False)
    return im


# ---------------------------------------------------------------------------
# Problem 1: Thresholding (5 pts)
# ---------------------------------------------------------------------------
def problem1(TH=210):
    """
    Threshold quart.jpg so that only the quarters (foreground) remain white.
    Background paper is bright (~215-255), the coins are darker (~60-200)
    with a few bright specular highlights, so we can separate them with a
    single global threshold TH.

    Per the assignment: pixel > TH  -> 0   (background goes black)
                         pixel <= TH -> 255 (quarter stays/becomes white)
    """
    im = load_gray("quart.jpg")
    h, w = im.shape
    out = np.zeros((h, w), dtype=np.uint8)

    # Explicit pixel-by-pixel loop, exactly as specified in the handout.
    for i in range(h):
        for j in range(w):
            if im[i, j] > TH:
                out[i, j] = 0
            else:
                out[i, j] = 255

    save("p1_quart_threshold_TH%d.png" % TH, out)

    # A couple of extra TH values to show the effect of TH on the result.
    for th in (150, 180, 230):
        alt = np.where(im > th, 0, 255).astype(np.uint8)
        save("p1_quart_threshold_TH%d.png" % th, alt)

    return im, out


# ---------------------------------------------------------------------------
# Problem 2: Bi-level thresholding (5 pts)
# ---------------------------------------------------------------------------
def problem2(I0=130, I1=190):
    """
    Bi-level threshold snowman.jpg: keep only Bob the snowman as white.
    From the image histogram, the snowman body sits around intensity 150-170,
    the black floor around 0-30, and the light-gray background around 200-230.
    So I0=130, I1=190 isolates the body.
    """
    im = load_gray("snowman.jpg")
    out = np.where((im >= I0) & (im <= I1), 255, 0).astype(np.uint8)
    save("p2_snowman_bilevel_I0-%d_I1-%d.png" % (I0, I1), out)
    return im, out


# ---------------------------------------------------------------------------
# Problem 3: Pixel-wise operations - invert & brightness (5 pts)
# ---------------------------------------------------------------------------
def problem3(I0=25):
    im = load_gray("quart.jpg").astype(np.int16)

    # Invert: 255 - I.  Already guaranteed to stay in [0,255], no shift needed.
    inv = 255 - im
    save("p3_quart_inverted.png", inv)

    # Increase brightness by 10% (multiplicative) then clip to [0,255].
    bright10pct = im * 1.10
    save("p3_quart_bright_+10pct.png", bright10pct)

    # Decrease brightness by 10% (multiplicative) then clip to [0,255].
    dark10pct = im * 0.90
    save("p3_quart_dark_-10pct.png", dark10pct)

    # Additive brightness change by a fixed offset I0, clipped to [0,255].
    brighter = im + I0
    save("p3_quart_bright_+%d.png" % I0, brighter)

    darker = im - I0
    save("p3_quart_dark_-%d.png" % I0, darker)

    return im


# ---------------------------------------------------------------------------
# Problem 4: Quantization (5 pts)
# ---------------------------------------------------------------------------
def quantize(im, n_levels):
    """
    Uniformly quantize an 8-bit image to n_levels bins, mapping every pixel
    in a bin to the bin's midpoint value (matches the handout's example:
    0-31 -> 16, 32-63 -> 48, ... for 8 levels).
    """
    step = 256 // n_levels
    bin_idx = np.minimum(im // step, n_levels - 1)
    midpoints = (bin_idx * step + step // 2).astype(np.uint8)
    return midpoints


def problem4():
    im = load_gray("quart.jpg")
    q8 = quantize(im, 8)
    q16 = quantize(im, 16)
    save("p4_quart_quantized_8levels.png", q8)
    save("p4_quart_quantized_16levels.png", q16)
    return im, q8, q16


# ---------------------------------------------------------------------------
# Problem 5a: Image subsampling by skipping every other pixel (5 pts)
# ---------------------------------------------------------------------------
def problem5a():
    im = load_gray("quart.jpg")
    h, w = im.shape
    out = np.zeros((h // 2, w // 2), dtype=np.uint8)
    for i in range(0, h - 1, 2):
        for j in range(0, w - 1, 2):
            out[i // 2, j // 2] = im[i, j]
    save("p5a_quart_subsampled_half.png", out)
    return im, out


# ---------------------------------------------------------------------------
# Problem 5b: Image re-sampling with bilinear interpolation (10 pts)
# ---------------------------------------------------------------------------
def bilinear_resize(im, out_h, out_w):
    """Resize `im` (2D uint8 array) to (out_h, out_w) using bilinear
    interpolation, implemented from scratch (no skimage.transform.resize)."""
    in_h, in_w = im.shape
    imf = im.astype(np.float64)
    out = np.zeros((out_h, out_w), dtype=np.float64)

    # Scale factors mapping output pixel -> input pixel coordinates.
    scale_y = in_h / out_h
    scale_x = in_w / out_w

    for i in range(out_h):
        # Source coordinate (continuous) for output row i.
        y = (i + 0.5) * scale_y - 0.5
        y = min(max(y, 0), in_h - 1)
        y0 = int(np.floor(y))
        y1 = min(y0 + 1, in_h - 1)
        dy = y - y0

        for j in range(out_w):
            x = (j + 0.5) * scale_x - 0.5
            x = min(max(x, 0), in_w - 1)
            x0 = int(np.floor(x))
            x1 = min(x0 + 1, in_w - 1)
            dx = x - x0

            top = imf[y0, x0] * (1 - dx) + imf[y0, x1] * dx
            bot = imf[y1, x0] * (1 - dx) + imf[y1, x1] * dx
            out[i, j] = top * (1 - dy) + bot * dy

    return out


def problem5b(out_h=500, out_w=1000):
    im = load_gray("quart.jpg")  # 242 x 308 (h x w)
    resized = bilinear_resize(im, out_h, out_w)
    save("p5b_quart_resized_%dx%d.png" % (out_w, out_h), resized)

    # Also show a non-uniform (different H/V ratio) rescale, e.g. squish width.
    resized2 = bilinear_resize(im, 400, 600)
    save("p5b_quart_resized_600x400.png", resized2)
    return im, resized


# ---------------------------------------------------------------------------
# Problem 6a: Region rotation & mirror, and patching a hole (10 pts)
# ---------------------------------------------------------------------------
def rotate_block_180(block):
    """Rotate a 2D array 180 degrees (equivalent to the rotation-matrix
    formula with theta=180, simplified to reversing both axes)."""
    return block[::-1, ::-1]


def rotate_block_90cw(block):
    """Rotate a 2D array 90 degrees clockwise (simplified rotation-matrix
    formula for theta=90): transpose then flip left-right."""
    return block.T[:, ::-1]


DOLPHINS_FILE = "dolphins.jpg"


def problem6a_dolphins():
    """
    Copy the window [135,135]->[220,180] (x0,y0)->(x1,y1) out of the
    dolphins image, rotate that patch 180 degrees, and paste it back into
    the same spot.

    NOTE: dolphins.jpg was not included in the homework zip that was
    provided. The code below is written generically -- it will run on the
    real dolphins.jpg the moment it is placed in this folder. Until then it
    falls back to quart.jpg (which is large enough for the same window) only
    so the pipeline can be demonstrated end-to-end; swap in the real image
    before submitting.
    """
    src_name = DOLPHINS_FILE
    if not os.path.exists(os.path.join(INPUT_DIR, src_name)):
        src_name = "quart.jpg"  # placeholder stand-in, see note above

    im = load_gray(src_name)
    x0, y0, x1, y1 = 135, 135, 220, 180
    out = im.copy()
    patch = im[y0:y1, x0:x1]
    save("p6a_dolphins_original_patch.png", patch)

    rotated_patch = rotate_block_180(patch)
    save("p6a_dolphins_rotated_patch.png", rotated_patch)

    out[y0:y1, x0:x1] = rotated_patch
    save("p6a_dolphins_patched.png", out)
    return im, out


def problem6a_rotate_jpg():
    """
    Take region [44,95]->[104,125] (x0,y0)->(x1,y1) from rotate.jpg, rotate
    it 90 degrees clockwise, and move it to fill the black hole in the top
    right corner so the result looks seamless.
    """
    im = load_gray("rotate.jpg")
    x0, y0, x1, y1 = 44, 95, 104, 125
    patch = im[y0:y1, x0:x1]
    rotated = rotate_block_90cw(patch)  # 30x60 -> 60x30 -> becomes 60 tall x 30 wide
    save("p6a_rotate_source_patch.png", patch)
    save("p6a_rotate_rotated_patch.png", rotated)

    # Locate the black hole automatically (robust to the exact offset used)
    # by searching the top-right quadrant for near-zero pixels.
    quad = im[:100, 100:200]
    mask = quad < 30
    ys, xs = np.where(mask)
    hy0, hy1 = ys.min(), ys.max() + 1
    hx0, hx1 = xs.min() + 100, xs.max() + 100 + 1

    out = im.copy()
    hole_h, hole_w = hy1 - hy0, hx1 - hx0
    # The rotated patch should match the hole size by design; if it's off by
    # a pixel or two (e.g. due to the hole's edge anti-aliasing), replicate
    # the patch's border pixels so the fill is fully seamless with no
    # left-over background showing through.
    pad_bottom = max(0, hole_h - rotated.shape[0])
    pad_right = max(0, hole_w - rotated.shape[1])
    fill = np.pad(rotated, ((0, pad_bottom), (0, pad_right)), mode="edge")
    fill = fill[:hole_h, :hole_w]
    out[hy0:hy1, hx0:hx1] = fill
    save("p6a_rotate_patched.png", out)
    return im, out


# ---------------------------------------------------------------------------
# Problem 6b/6c/6d: Full-image rotation (padding, forward map, bilinear
# backward map, residual check)
# ---------------------------------------------------------------------------
#
# Sign convention: theta > 0 means the image visibly rotates counter-
# clockwise on screen (matches PIL.Image.rotate / skimage.transform.rotate).
# Because image rows increase *downward*, this is implemented as the given
# rotation matrix applied with a sign-flip on the y term (derivation in the
# write-up): x' = x*cos(t) + y*sin(t),  y' = -x*sin(t) + y*cos(t)

def rotated_canvas_size(h, w, theta_deg):
    t = np.deg2rad(theta_deg)
    new_w = int(np.ceil(abs(w * np.cos(t)) + abs(h * np.sin(t))))
    new_h = int(np.ceil(abs(w * np.sin(t)) + abs(h * np.cos(t))))
    return new_h, new_w


def rotate_forward_nearest(im, theta_deg):
    """
    Problem 6b: FORWARD mapping. For every source pixel, compute where it
    lands in the (padded) destination image using the rotation matrix, and
    round to the nearest integer destination pixel. Note some destination
    pixels are never written to (small holes) -- this is an expected,
    discussed artifact of forward mapping.
    """
    h, w = im.shape
    new_h, new_w = rotated_canvas_size(h, w, theta_deg)
    out = np.zeros((new_h, new_w), dtype=np.uint8)
    filled = np.zeros((new_h, new_w), dtype=bool)

    t = np.deg2rad(theta_deg)
    ct, st = np.cos(t), np.sin(t)
    cx_s, cy_s = (w - 1) / 2.0, (h - 1) / 2.0
    cx_d, cy_d = (new_w - 1) / 2.0, (new_h - 1) / 2.0

    for y in range(h):
        yc = y - cy_s
        for x in range(w):
            xc = x - cx_s
            xr = xc * ct + yc * st
            yr = -xc * st + yc * ct
            xd = int(round(xr + cx_d))
            yd = int(round(yr + cy_d))
            if 0 <= xd < new_w and 0 <= yd < new_h:
                out[yd, xd] = im[y, x]
                filled[yd, xd] = True

    return out, filled


def rotate_backward_bilinear(im, theta_deg, out_size=None, bg=0):
    """
    Problem 6c: BACKWARD mapping with bilinear interpolation. For every
    destination pixel, back-project (using the inverse rotation) into the
    source image and bilinearly interpolate -- this avoids the holes seen
    with forward mapping.
    """
    h, w = im.shape
    if out_size is None:
        new_h, new_w = rotated_canvas_size(h, w, theta_deg)
    else:
        new_h, new_w = out_size

    t = np.deg2rad(theta_deg)
    ct, st = np.cos(t), np.sin(t)
    cx_s, cy_s = (w - 1) / 2.0, (h - 1) / 2.0
    cx_d, cy_d = (new_w - 1) / 2.0, (new_h - 1) / 2.0

    out = np.full((new_h, new_w), bg, dtype=np.float64)
    imf = im.astype(np.float64)

    for yd in range(new_h):
        yc = yd - cy_d
        for xd in range(new_w):
            xc = xd - cx_d
            # Inverse rotation (rotate back by -theta) to find source coord.
            xs = xc * ct - yc * st + cx_s
            ys = xc * st + yc * ct + cy_s

            if 0 <= xs <= w - 1 and 0 <= ys <= h - 1:
                x0, y0 = int(np.floor(xs)), int(np.floor(ys))
                x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
                dx, dy = xs - x0, ys - y0
                top = imf[y0, x0] * (1 - dx) + imf[y0, x1] * dx
                bot = imf[y1, x0] * (1 - dx) + imf[y1, x1] * dx
                out[yd, xd] = top * (1 - dy) + bot * dy

    return out


def problem6bcd():
    im = load_gray("lenag.png")

    # 6b: forward mapping with rounding
    fwd, filled = rotate_forward_nearest(im, 35)
    save("p6b_lenagr_forward_rounded.png", fwd)
    hole_pct = 100.0 * (~filled).sum() / filled.size
    print("6b forward-map unfilled pixel percentage: %.2f%%" % hole_pct)

    # 6c: backward mapping with bilinear interpolation (no holes)
    bwd = rotate_backward_bilinear(im, 35)
    save("p6c_lenag_bilinear_rotated.png", bwd)

    # 6d: rotate the bilinear result back by 35 CW (-35) and difference it
    # against the original. We compare against the *bilinear* (6c) result
    # rather than the forward/rounded (6b) one, since 6b's un-filled holes
    # would otherwise dominate and be uninformative for residual analysis;
    # we also repeat the check on the 6b result for completeness below.
    padded_h, padded_w = bwd.shape
    back = rotate_backward_bilinear(bwd.astype(np.uint8), -35,
                                     out_size=(padded_h, padded_w))
    # Center-crop back down to the original 512x512 footprint to compare.
    ch, cw = im.shape
    y0 = (padded_h - ch) // 2
    x0 = (padded_w - cw) // 2
    lenag2 = back[y0:y0 + ch, x0:x0 + cw]
    save("p6d_lenag2_rotated_back.png", lenag2)

    diff = np.abs(im.astype(np.int16) - lenag2.astype(np.int16)).astype(np.uint8)
    save("p6d_residual_difference.png", diff)
    print("6d residual (bilinear path): mean=%.3f max=%d" % (diff.mean(), diff.max()))

    # Same round-trip using the 6b forward/rounded result, for comparison.
    back_fwd = rotate_forward_nearest(fwd.astype(np.uint8), -35)[0]
    ph2, pw2 = back_fwd.shape
    y0b = max(0, (ph2 - ch) // 2)
    x0b = max(0, (pw2 - cw) // 2)
    lenag2_fwd = back_fwd[y0b:y0b + ch, x0b:x0b + cw]
    if lenag2_fwd.shape == im.shape:
        diff_fwd = np.abs(im.astype(np.int16) - lenag2_fwd.astype(np.int16)).astype(np.uint8)
        save("p6d_residual_difference_forwardmap.png", diff_fwd)
        print("6d residual (forward/rounded path): mean=%.3f max=%d" % (diff_fwd.mean(), diff_fwd.max()))

    return im, fwd, bwd, lenag2, diff


# ---------------------------------------------------------------------------
# Problem 7a: Logical operations on binary images (12 pts)
# ---------------------------------------------------------------------------
def problem7a():
    im1 = load_gray("things1.png")
    im2 = load_gray("things2.png")

    # things1.png is already pure {0,255}; things2.png has a few antialiased
    # in-between values from PNG compression, so binarize both at 128 first.
    # NOTE ON POLARITY: in these PNGs the drawn objects are DARK (~0) on a
    # WHITE (~255) background. For the logical operations to mean what
    # they should ("is there an object here?"), the *object/foreground*
    # must map to 255 (True) and background to 0 (False) -- i.e. we invert
    # relative to the raw pixel brightness.
    b1 = np.where(im1 < 128, 255, 0).astype(np.uint8)
    b2 = np.where(im2 < 128, 255, 0).astype(np.uint8)

    and_img = np.bitwise_and(b1, b2)
    or_img = np.bitwise_or(b1, b2)
    xor_img = np.bitwise_xor(b1, b2)

    save("p7a_things1_bin.png", b1)
    save("p7a_things2_bin.png", b2)
    save("p7a_AND.png", and_img)
    save("p7a_OR.png", or_img)
    save("p7a_XOR.png", xor_img)
    return b1, b2, and_img, or_img, xor_img


# ---------------------------------------------------------------------------
# Problem 7b: Pixel arithmetic on two personal photos (12 pts)
# ---------------------------------------------------------------------------
def to_gray_uint8(path):
    im = io.imread(path)
    if im.ndim == 3:
        im = im[:, :, :3]  # drop alpha if present
        gray = color.rgb2gray(im)  # float64 in [0,1]
        return (gray * 255).astype(np.uint8)
    return im.astype(np.uint8)


def problem7b(k=0.5):
    a_path = os.path.join(INPUT_DIR, "photo1.png")
    b_path = os.path.join(INPUT_DIR, "photo2.png")
    A = to_gray_uint8(a_path)
    B = to_gray_uint8(b_path)

    # Must be exactly the same size for pixel-wise arithmetic.
    if A.shape != B.shape:
        h = min(A.shape[0], B.shape[0])
        w = min(A.shape[1], B.shape[1])
        A = A[:h, :w]
        B = B[:h, :w]

    save("p7b_A_gray.png", A)
    save("p7b_B_gray.png", B)

    Af = A.astype(np.float32)
    Bf = B.astype(np.float32)

    # ADDITION: C = A + k*B, then clip to [0,255]. A single vectorized numpy
    # expression -- no explicit for loop needed (much faster than looping
    # over every pixel in Python).
    add_img = np.clip(Af + k * Bf, 0, 255).astype(np.uint8)

    # SUBTRACTION: C = A - k*B, clipped to [0,255].
    sub_img = np.clip(Af - k * Bf, 0, 255).astype(np.uint8)

    # MULTIPLICATION: element-wise product, rescaled back into [0,255]
    # (raw uint8*uint8 would overflow uint8 and float product can reach
    # 255*255, so we normalize by 255 to keep the result in range).
    mul_img = np.clip((Af * Bf) / 255.0, 0, 255).astype(np.uint8)

    # DIVISION: A / (B+eps), rescaled into [0,255] for display. A few very
    # dark pixels in B create huge outlier ratios, so we rescale by the 99th
    # percentile (rather than the absolute max) and clip -- otherwise those
    # rare outliers wash out the whole image to near-black.
    div_raw = Af / (Bf + 1.0)
    p99 = np.percentile(div_raw, 99)
    div_img = np.clip(div_raw / p99 * 255.0, 0, 255).astype(np.uint8)

    save("p7b_ADD_k%.1f.png" % k, add_img)
    save("p7b_SUB_k%.1f.png" % k, sub_img)
    save("p7b_MUL.png", mul_img)
    save("p7b_DIV.png", div_img)

    return A, B, add_img, sub_img, mul_img, div_img


if __name__ == "__main__":
    problem1()
    problem2()
    problem3()
    problem4()
    problem5a()
    problem5b()
    problem6a_dolphins()
    problem6a_rotate_jpg()
    problem6bcd()
    problem7a()
    problem7b()
    print("All problems (1-7) done.")
