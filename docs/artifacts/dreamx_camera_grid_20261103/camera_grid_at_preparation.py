"""Resample a fixed camera centre onto a declared pinhole pixel grid.

This changes intrinsics and field of view, never camera pose or depth units.
It is preprocessing from the input camera contract, not trajectory calibration.
"""
from dataclasses import dataclass
import numpy as np
from scipy.ndimage import map_coordinates


def _intrinsics(value):
    k=np.asarray(value,dtype=float)
    if (k.shape!=(3,3) or not np.isfinite(k).all() or k[0,0]<=0 or k[1,1]<=0
            or not np.allclose(k[2],[0,0,1],rtol=0,atol=1e-12)
            or abs(k[1,0])>1e-12):
        raise ValueError('Expected finite pinhole intrinsics with positive focal lengths')
    return k.copy()


@dataclass(frozen=True)
class CameraGrid:
    """Integer pixel centres, OpenCV rays, unchanged optical axis and centre.

    Sizes are (width,height). Outside-source RGB is black with a separate mask;
    unavailable depth stays NaN. Callers must check coverage before generation.
    """
    source_size: tuple[int,int]
    output_size: tuple[int,int]
    source_K: np.ndarray
    output_K: np.ndarray

    def __post_init__(self):
        for size in [self.source_size,self.output_size]:
            if (len(size)!=2 or any(isinstance(x,(bool,np.bool_))
                    or not isinstance(x,(int,np.integer)) or x<=0 for x in size)):
                raise ValueError('Pixel-grid dimensions must be positive integers')
        for name in ['source_K','output_K']:
            k=_intrinsics(getattr(self,name));k.setflags(write=False)
            object.__setattr__(self,name,k)

    def coordinates(self):
        w,h=self.output_size;sw,sh=self.source_size
        yy,xx=np.mgrid[:h,:w]
        pixels=np.stack([xx,yy,np.ones_like(xx)],axis=0).reshape(3,-1)
        xy=(self.source_K@np.linalg.inv(self.output_K)@pixels)[:2].reshape(2,h,w)
        valid=(xy[0]>=0)&(xy[0]<=sw-1)&(xy[1]>=0)&(xy[1]<=sh-1)
        return xy,valid

    def rgb(self,image):
        image=np.asarray(image)
        if image.shape!=(*self.source_size[::-1],3) or image.dtype!=np.uint8:
            raise ValueError('Expected source-grid uint8 RGB')
        xy,valid=self.coordinates()
        out=np.stack([map_coordinates(image[...,c].astype(float),[xy[1],xy[0]],
                      order=1,mode='constant',cval=0,prefilter=False) for c in range(3)],axis=-1)
        return np.rint(out).clip(0,255).astype(np.uint8),valid

    def depth(self,depth):
        depth=np.asarray(depth,dtype=float)
        if depth.shape!=self.source_size[::-1]:
            raise ValueError('Expected source-grid optical-axis z depth')
        xy,in_view=self.coordinates();valid=np.isfinite(depth)&(depth>0)
        coords=[xy[1],xy[0]]
        values=map_coordinates(np.where(valid,depth,0),coords,order=1,
                               mode='constant',cval=0,prefilter=False)
        weights=map_coordinates(valid.astype(float),coords,order=1,
                                mode='constant',cval=0,prefilter=False)
        supported=in_view&(weights>=1-1e-7)
        # No pose rotation: z depth is sampled on the new ray, not rescaled.
        out=np.full(self.output_size[::-1],np.nan,dtype=float)
        out[supported]=values[supported]/weights[supported]
        return out,supported
