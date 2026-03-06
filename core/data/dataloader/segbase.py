"""Base segmentation dataset"""
import random
import numpy as np
import cv2

from PIL import Image, ImageOps, ImageFilter
Image.MAX_IMAGE_PIXELS = None 

__all__ = ['SegmentationDataset']


class SegmentationDataset(object):
    """Segmentation Base Dataset"""

    def __init__(self, root, split, mode, transform, base_size=520, crop_size=480):
        super(SegmentationDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.split = split
        self.mode = mode if mode is not None else split
        self.base_size = base_size
        self.crop_size = crop_size

    def _val_sync_transform(self, img, mask):

        
        outsize = self.crop_size
        short_size = outsize
        w, h = img.size
        if w > h:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - outsize) / 2.))
        y1 = int(round((h - outsize) / 2.))
        img = img.crop((x1, y1, x1 + outsize, y1 + outsize))
        mask = mask.crop((x1, y1, x1 + outsize, y1 + outsize))
        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask
        
        
        '''
        w, h = img.size
        target_w = 0
        target_h = 0

        if w < 2048 and h < 2048:
           target_w, target_h = 2048, 2048
        elif w < 2048:
           target_w, target_h = 2048, h
        elif h < 2048:
           target_w, target_h = w, 2048
        else:
           target_w, target_h = w, h
        # already big enough, return unchanged
        #   return img, mask

        pad_w = target_w - w
        pad_h = target_h - h
        # symmetric padding: half on each side
        left = pad_w // 2
        right = pad_w - left
        top = pad_h // 2
        bottom = pad_h - top

        img_p = ImageOps.expand(img, border=(left, top, right, bottom), fill=0)
        mask_p = ImageOps.expand(mask, border=(left, top, right, bottom), fill=0)

        img = img_p.resize((1024, 1024), Image.BILINEAR)
        mask = mask_p.resize((1024, 1024), Image.NEAREST)
        

        return self._img_transform(img), self._mask_transform(mask)
        '''

        


        
        

        
        '''
        w, h = img.size
        if w != 1024 or h != 1024:
           img = img.resize((1024, 1024), Image.BILINEAR)
           mask = mask.resize((1024, 1024), Image.NEAREST)
        
        #img = img.resize((512, 512), Image.BILINEAR)
        #mask = mask.resize((512, 512), Image.NEAREST)
        
        return self._img_transform(img), self._mask_transform(mask)
        '''
        








    def _sync_transform(self, img, mask):
        
        # random mirror
         # random mirror
        #if random.random() < 0.5:
        #    img = img.transpose(Image.FLIP_LEFT_RIGHT)
        #    mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        
        
        crop_size = self.crop_size
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        # gaussian blur as in PSP
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))
        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask
        

        '''
        w, h = img.size
       
        if w < 2048 and h < 2048:
           target_w, target_h = 2048, 2048
        elif w < 2048:
           target_w, target_h = 2048, h
        elif h < 2048:
           target_w, target_h = w, 2048
        else:
           target_w, target_h = w, h
        # already big enough, return unchanged
        #   return img, mask

        pad_w = target_w - w
        pad_h = target_h - h
        # symmetric padding: half on each side
        left = pad_w // 2
        right = pad_w - left
        top = pad_h // 2
        bottom = pad_h - top

        img_p = ImageOps.expand(img, border=(left, top, right, bottom), fill=0)
        mask_p = ImageOps.expand(mask, border=(left, top, right, bottom), fill=0)

        img = img_p.resize((1024, 1024), Image.BILINEAR)
        mask = mask_p.resize((1024, 1024), Image.NEAREST)

        return self._img_transform(img), self._mask_transform(mask)
        '''


        


        '''
        w, h = img.size
        if w != 1024 or h != 1024:
           img = img.resize((1024, 1024), Image.BILINEAR)
           mask = mask.resize((1024, 1024), Image.NEAREST)
        
        
        
        return self._img_transform(img), self._mask_transform(mask)
        '''
        
        #Process w
        
    #Train full image:
      
       
    def _img_transform(self, img):
        return np.array(img)

    def _mask_transform(self, mask):
        return np.array(mask).astype('int32')

    @property
    def num_class(self):
        """Number of categories."""
        return self.NUM_CLASS

    @property
    def pred_offset(self):
        return 0






'''

class RandomTidyLongSideCrop:
    """
    Pair-wise (image, mask) transform for semantic segmentation on large grayscale images.

    Steps:
      1) (Optional) Tidy: crop to bbox of non-black pixels on the IMAGE; apply same bbox to MASK.
      2) Resize so the LONGER side is in [min_long, max_long] (sampled each call).
      3) Pad (reflect or constant) if needed, then random crop to crop_size.
      4) Light grayscale-safe augs (flip/rot/blur/gamma). Mask uses NEAREST everywhere.

    Notes:
      - Image: BOX→LANCZOS path when shrinking a lot, else LANCZOS once.
      - Mask: always NEAREST to preserve class IDs.
      - Works with image.mode 'L'. If your masks are palettes/uint8 IDs, leave them as-is.
    """
    def __init__(
        self,
        crop_size: int,
        min_long: int = 1024,
        max_long: int = 2048,
        content_crop: bool = True,
        nonblack_threshold: int = 1,
        safety_margin: int = 0,
        pad_mode: str = "reflect",   # "reflect" or "constant"
        pad_value_img: int = 0,
        pad_value_mask: int = 0,
        hflip_p: float = 0.5,
        vflip_p: float = 0.0,
        max_rotate_deg: float = 0.0, # e.g., 3.0 for slight rotations
        blur_p: float = 0.3,
        gamma_p: float = 0.3,
        gamma_range: Tuple[float, float] = (0.9, 1.1),
        to_tensor=lambda x: TF.to_tensor(x),        # override with your pipeline’s
        mask_to_tensor=lambda x: TF.pil_to_tensor(x) # preserves integer labels
    ):
        assert min_long <= max_long
        self.crop_size = crop_size
        self.min_long = min_long
        self.max_long = max_long
        self.content_crop = content_crop
        self.nonblack_threshold = nonblack_threshold
        self.safety_margin = safety_margin
        self.pad_mode = pad_mode
        self.pad_value_img = pad_value_img
        self.pad_value_mask = pad_value_mask
        self.hflip_p = hflip_p
        self.vflip_p = vflip_p
        self.max_rotate_deg = max_rotate_deg
        self.blur_p = blur_p
        self.gamma_p = gamma_p
        self.gamma_range = gamma_range
        self._to_tensor = to_tensor
        self._mask_to_tensor = mask_to_tensor

    # ---------- utilities ----------
    @staticmethod
    def _find_nonblack_bbox_gray(img_gray: Image.Image, thr: int) -> Optional[Tuple[int,int,int,int]]:
        g = img_gray.convert("L")
        arr = np.asarray(g, dtype=np.uint8)
        mask = arr > thr
        if not mask.any():
            return None
        ys, xs = np.nonzero(mask)
        l, r = xs.min(), xs.max()
        t, b = ys.min(), ys.max()
        return (l, t, r + 1, b + 1)  # right/bottom exclusive

    @staticmethod
    def _resize_long_side_quality(img: Image.Image, target_long: int, is_mask: bool) -> Image.Image:
        w, h = img.size
        long_side = max(w, h)
        if long_side == target_long:
            return img
        scale = target_long / long_side
        tw, th = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        if is_mask:
            return img.resize((tw, th), Image.NEAREST)

        # quality path for images
        if long_side > 2.2 * target_long:
            inter = (max(tw * 3 // 2, tw + 1), max(th * 3 // 2, th + 1))
            img = img.resize(inter, Image.BOX)
        return img.resize((tw, th), Image.LANCZOS)

    def _pad_to_min(self, img: Image.Image, msk: Image.Image, min_w: int, min_h: int):
        w, h = img.size
        pad_w = max(0, min_w - w)
        pad_h = max(0, min_h - h)
        if pad_w == 0 and pad_h == 0:
            return img, msk
        if self.pad_mode == "reflect":
            # reflect pad via expand+crop trick
            img_p = ImageOps.expand(img, border=(0,0,pad_w,pad_h), fill=0)
            msk_p = ImageOps.expand(msk, border=(0,0,pad_w,pad_h), fill=self.pad_value_mask)
            # emulate reflect by mirroring the needed border regions
            if pad_w or pad_h:
                # Right/bottom fill by flipped slices
                # (Simple, effective for small pads; for big pads, switch to constant)
                img_p = self._mirror_fill(img_p, w, h)
                msk_p = self._mirror_fill(msk_p, w, h)
            return img_p, msk_p
        else:
            img_p = ImageOps.expand(img, border=(0,0,pad_w,pad_h), fill=self.pad_value_img)
            msk_p = ImageOps.expand(msk, border=(0,0,pad_w,pad_h), fill=self.pad_value_mask)
            return img_p, msk_p

    @staticmethod
    def _mirror_fill(im: Image.Image, orig_w: int, orig_h: int) -> Image.Image:
        w, h = im.size
        if w > orig_w:
            right = im.crop((orig_w - (w - orig_w), 0, orig_w, h)).transpose(Image.FLIP_LEFT_RIGHT)
            im.paste(right, (orig_w, 0))
        if h > orig_h:
            bottom = im.crop((0, orig_h - (h - orig_h), w, orig_h)).transpose(Image.FLIP_TOP_BOTTOM)
            im.paste(bottom, (0, orig_h))
        return im

    # ---------- main call ----------
    def __call__(self, img: Image.Image, mask: Image.Image):
        # 0) enforce grayscale for bbox logic
        assert img.mode in ("L", "I;16", "F") or True  # we convert below when needed
        # 1) content crop (optional)
        if self.content_crop:
            bbox = self._find_nonblack_bbox_gray(img, self.nonblack_threshold)
            if bbox is not None:
                l,t,r,b = bbox
                if self.safety_margin > 0:
                    W, H = img.size
                    l = max(0, l - self.safety_margin)
                    t = max(0, t - self.safety_margin)
                    r = min(W, r + self.safety_margin)
                    b = min(H, b + self.safety_margin)
                img = img.crop((l,t,r,b))
                mask = mask.crop((l,t,r,b))

        # 2) sample target long side (log-uniform is nicer for scale diversity)
        log_min, log_max = math.log(self.min_long), math.log(self.max_long)
        target_long = int(round(math.exp(random.uniform(log_min, log_max))))

        img = self._resize_long_side_quality(img, target_long, is_mask=False)
        mask = self._resize_long_side_quality(mask, target_long, is_mask=True)

        # 3) pad to at least crop_size, then random crop
        img, mask = self._pad_to_min(img, mask, self.crop_size, self.crop_size)
        w, h = img.size
        # guard if still smaller (rare)
        if w < self.crop_size or h < self.crop_size:
            pad_w = max(0, self.crop_size - w)
            pad_h = max(0, self.crop_size - h)
            img = ImageOps.expand(img, border=(0,0,pad_w,pad_h), fill=0)
            mask = ImageOps.expand(mask, border=(0,0,pad_w,pad_h), fill=self.pad_value_mask)
            w, h = img.size

        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        # 4) paired photometric/geom augs (label-safe)
        if self.hflip_p > 0 and random.random() < self.hflip_p:
            img = TF.hflip(img); mask = TF.hflip(mask)
        if self.vflip_p > 0 and random.random() < self.vflip_p:
            img = TF.vflip(img); mask = TF.vflip(mask)
        if self.max_rotate_deg > 0:
            deg = random.uniform(-self.max_rotate_deg, self.max_rotate_deg)
            img = TF.rotate(img, deg, interpolation=TF.InterpolationMode.BILINEAR, fill=0)
            mask = TF.rotate(mask, deg, interpolation=TF.InterpolationMode.NEAREST, fill=self.pad_value_mask)
        if self.blur_p > 0 and random.random() < self.blur_p:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.5)))
        if self.gamma_p > 0 and random.random() < self.gamma_p:
            g = random.uniform(*self.gamma_range)
            # TF.adjust_gamma expects gamma>0; for grayscale, it’s safe.
            img = TF.adjust_gamma(img, gamma=g, gain=1.0)

        # 5) final to-tensor conversions (you can plug your own normalizer)
        img_t = self._to_tensor(img)          # float tensor [0,1] or normalized
        mask_t = self._mask_to_tensor(mask)   # int/long labels (no one-hot here)

        return img_t, mask_t
'''



