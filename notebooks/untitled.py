import xarray as xr
import numpy as np
from skimage.transform import hough_line, hough_line_peaks
from skimage.draw import line
import skimage.measure as skim
from tqdm.notebook import trange

import matplotlib.pyplot as plt
from matplotlib import cm

@xr.register_dataarray_accessor("imP")


class imP(object):
    '''
    This is a class to performed image analysis
    '''
    
    def __init__(self, xarray_obj):
        '''
        Constructor for imP
        '''
        self._obj = xarray_obj 
    pass


    def auto_correlation(self,pad=1,method='nearest',dic=False,fill_inf=True):
        '''
        :param pad: pading between 1 and 2
        :type pad: float
        :param method: method option for xr.DataArray.sel
        :param dic: use 1-(1-Cinf)/2 for extracting with at half height for speckle caracteriastion
        :type dic: bool
        :param fill_inf: fill correlation length largeur than the image with the imag length in the given direction
        :type fill_inf: bool
        '''
        data=np.array(self._obj)
        # Compute mean for replacing nan value and for padding
        mean_data=np.nanmean(data)
        # Replace nan value
        id=np.where(np.isnan(data))
        data[id]=mean_data
        if pad >1:
            ss=data.shape
            fpad=np.ones([pad*ss[0],pad*ss[1]])*mean_data
            fpad[0:ss[0],0:ss[1]]=data
        else:
            fpad=data

        # Compute autocorrelation
        FFT_fpad=np.fft.fft2(fpad)
        abs_FFTpad=np.abs(FFT_fpad)**2
        An=np.fft.ifft2(abs_FFTpad)
        mAn=np.nanmax(An)

        Autocor=np.abs(np.fft.fftshift(An/mAn));
        Cinf=mean_data**2/np.mean(fpad**2);
        res=xr.DataArray(Autocor,dims=('ya','xa'))
        ssa=Autocor.shape
        d_dims=self._obj.dims
        res.coords['ya']=np.array(np.abs(self._obj[d_dims[1]][1]-self._obj[d_dims[1]][0]))*np.linspace(0,ssa[0]-1,ssa[0])
        res.coords['xa']=np.array(np.abs(self._obj[d_dims[0]][1]-self._obj[d_dims[0]][0]))*np.linspace(0,ssa[1]-1,ssa[1])


        # etract min an max direction
        ss=res.shape

        x0=np.array(ss[1]/2*res.xa[1])
        y0=np.array(ss[0]/2*res.ya[1])
        cross=np.ones(180)

        ds=xr.Dataset()

        ds['AutoCorrelation']=res
        if dic:
            ds['Cinf']=(1+Cinf)/2
        else:
            ds['Cinf']=Cinf

        lmax=np.zeros(180)
        lmax_int=np.zeros(180)
        
        # compute length vs angle
        angle=np.linspace(0,89,90)*np.pi/180
        lmm=np.zeros(90)
        lmm[np.tan(angle)<y0/x0]=x0*(1+np.tan(angle[np.tan(angle)<y0/x0])**2)**0.5
        lmm[np.tan(angle)>y0/x0]=y0*(1+1/np.tan(angle[np.tan(angle)>y0/x0])**2)**0.5
        lmmfull=np.concatenate([lmm,lmm[::-1]])

        for i in list(range(180)):
            xt=x0+np.cos(i*np.pi/180.)*lmmfull[i]
            yt=y0+np.sin(i*np.pi/180.)*lmmfull[i]

            nb=int(((y0-yt)**2+(x0-xt)**2)**0.5/np.abs(self._obj[d_dims[0]][1]-self._obj[d_dims[0]][0]))
            xx=xr.DataArray(np.linspace(x0,xt,nb), dims="d"+str(i))
            yy=xr.DataArray(np.linspace(y0,yt,nb), dims="d"+str(i))

            profil=res.sel(xa=xx,ya=yy, method=method,drop=True)
            d=((xx-xx[0])**2+(yy-yy[0])**2)**0.5
            profil.coords['d'+str(i)]=d
            profil.attrs['angle']=i

            ds['P'+str(i)]=profil

            id=np.where((ds['P'+str(i)]>ds.Cinf)==False)[0]
            if np.size(id)==0:
                lmax[i]=np.nan
            else:
                lmax[i]=ds['d'+str(i)][id[0]]

            A1=(profil-ds.Cinf).integrate(profil.dims[0])
            lmax_int[i]=np.array(2*A1/(1-ds.Cinf))
        
        if fill_inf:
            lmax[np.isnan(lmax)]=lmmfull[np.isnan(lmax)]
            
        dlmax=xr.DataArray(lmax,dims='angle')
        dlmax_int=xr.DataArray(lmax_int,dims='angle')
        dlmax.coords['angle']=np.linspace(0,179,180)
        dlmax.attrs['unit_angle']='degree'

        ds['lmax']=dlmax
        ds['lmax_int']=dlmax_int
        return ds

    def hough_transform(self,**kwarg):
        '''
        Peformed hough_tranform and detect line
        
        based on https://scikit-image.org/docs/stable/auto_examples/edges/plot_line_hough_transform.html
        '''



        # Constructing test image
        image = np.array(self._obj)

        # Classic straight-line Hough transform
        # Set a precision of 0.5 degree.
        tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
        h, theta, d = hough_line(image, theta=tested_angles)

        peak=hough_line_peaks(h, theta, d,**kwarg)
                
        # Generating figure 1
        fig, axes = plt.subplots(1, 3, figsize=(15, 6))
        ax = axes.ravel()

        ax[0].imshow(image, cmap=cm.gray)
        ax[0].set_title('Input image')
        ax[0].set_axis_off()

        angle_step = 0.5 * np.diff(theta).mean()
        d_step = 0.5 * np.diff(d).mean()
        bounds = [np.rad2deg(theta[0] - angle_step),
                  np.rad2deg(theta[-1] + angle_step),
                  d[-1] + d_step, d[0] - d_step]
        im1=ax[1].imshow(h, extent=bounds, cmap=cm.gray, aspect=1 / 1.5)
        ax[1].set_title('Hough transform')
        ax[1].set_xlabel('Angles (degrees)')
        ax[1].set_ylabel('Distance (pixels)')
        ax[1].axis('image')
        

        for i in range(len(peak[0])):
            angle = peak[1][i]*180/np.pi
            dist = peak[2][i]
            ax[1].plot(angle,dist,'or')

        im2=ax[2].imshow(image, cmap=cm.gray)
        ax[2].set_ylim((image.shape[0], 0))
        ax[2].set_axis_off()
        ax[2].set_title('Detected lines')

        # for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
        for i in range(len(peak[0])):
            angle = peak[1][i]
            dist = peak[2][i]
            (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
            ax[2].axline((x0, y0), slope=np.tan(angle + np.pi/2))

        plt.tight_layout()
        
        # add space for colour bar
        plt.show()
        
        return peak
    
    def circle_filter(self,vfill=np.nan):
        '''
        Crop the image into circle image in a square.
        '''
        im_np=np.array(self._obj)
        ss=np.min(np.shape(im_np))
        
        crop_im_np=im_np[0:ss,0:ss]
        
        x = np.linspace(-1, 1, ss)
        y = np.linspace(-1, 1, ss)
        xx, yy = np.meshgrid(x, y)

        zz = np.sqrt(xx**2 + yy**2)

        crop_im_np[np.where(zz>1)]=vfill
        
        return xr.DataArray(crop_im_np)
    
    def inertia_tensor(self):
        '''
        Compute the inertia tensor of the image and retrun the eigenvalue and eigenvector.
        '''
        inertia=skim.inertia_tensor(np.array(self._obj))
        eigval,eigvec=np.linalg.eig(inertia)
        
        ds=xr.Dataset()
        if eigval[0]>eigval[1]:
            ds['a1']=eigval[0]
            ds['e1']=eigvec[:,0]
            ds['a2']=eigval[1]
            ds['e2']=eigvec[:,1]
        else:
            ds['a1']=eigval[1]
            ds['e1']=eigvec[:,1]
            ds['a2']=eigval[0]
            ds['e2']=eigvec[:,0]
            
        return ds
    
    def texture_anisotropy(self,size_box,pix_shift=0,cutoff_sphere=False,store_sub_im=False):
        '''
        Compute the inertia tensor on all the image for sub image of size size_box
        '''
        res=self._obj[self._obj.dims[0]][1]-self._obj[self._obj.dims[0]][0]
        texture_anisotropy=np.zeros(len(size_box))
        all_dict=[]
        for bb in range(len(size_box)):
            bb_ellipse=[]
            bb_ani=[]
            rose_angle=[]
            rose_force=[]
            center_box,sub_img=self.split_img(size_box[bb],pix_shift=pix_shift)
            for i in range(len(center_box)):
                if cutoff_sphere:
                    u_im=sub_img[i].imP.circle_filter(vfill=0)
                else:
                    u_im=sub_img[i]
                # might not be the best when circle filter is on but in practice should not be a problem        
                if len(np.unique(u_im))!=1: # don't when to include uniform image in the data set
                    ds_ell=u_im.imP.inertia_tensor()
                    ds_ell['a1']=ds_ell.a1*res
                    ds_ell['a2']=ds_ell.a2*res
                    if store_sub_im:
                        ds_ell['sub_image']=u_im
                    ds_ell['center']=np.array(center_box[i])*np.float64(res)
                    ds_ell['box_size']=size_box[bb]*res
                    ds_ell['anisotropy']=1-ds_ell.a2/ds_ell.a1
                    ds_ell['angle']=np.arctan(ds_ell.e2[0]/ds_ell.e2[1])


                        
                    if ~np.isnan(ds_ell.anisotropy):
                        rose_force.append(ds_ell.anisotropy)
                        rose_angle.append(ds_ell.angle)
                    
                        bb_ellipse.append(ds_ell)
                        bb_ani.append(np.array(ds_ell.anisotropy))
            
            all_dict.append({"box_size":size_box[bb]*res,"image":self._obj,"ellipse":bb_ellipse,"rose_data":np.dstack([rose_angle,rose_force])[0]})
                
            texture_anisotropy[bb]=np.mean(bb_ani)
            
            
                        
        return np.array(size_box),texture_anisotropy,all_dict
    
    def texture_anisotropy_2(self,size_box,pix_shift=0,cutoff_sphere=False):
        '''
        Compute the inertia tensor on all the image for sub image of size size_box
        '''
        res=self._obj[self._obj.dims[0]][1]-self._obj[self._obj.dims[0]][0]
        texture_anisotropy=np.zeros(len(size_box))
        all_dict=[]
        coordinate=self._obj.coords.dims
        for bb in range(len(size_box)):
            ds_tmp=xr.Dataset()
            ds_tmp.attrs['box_size']=np.array(np.abs(size_box[bb]*res))
            
            ss=np.shape(self._obj)
            if pix_shift==0:
                xx=np.arange(0, ss[0], size_box[bb])
                yy=np.arange(0, ss[1], size_box[bb])
            else:
                xx=np.arange(0,ss[0]-size_box[bb],pix_shift)
                yy=np.arange(0,ss[1]-size_box[bb],pix_shift)
            
            ds_tmp['a1']=xr.DataArray(np.zeros([len(xx)-1,len(yy)-1]),dims=coordinate)
            ds_tmp['a2']=xr.DataArray(np.zeros([len(xx)-1,len(yy)-1]),dims=coordinate)
            ds_tmp['e1']=xr.DataArray(np.zeros([len(xx)-1,len(yy)-1,2]),dims=(coordinate[0],coordinate[1],'uvecs'))
            ds_tmp['e2']=xr.DataArray(np.zeros([len(xx)-1,len(yy)-1,2]),dims=(coordinate[0],coordinate[1],'uvecs'))
            ds_tmp['angle']=xr.DataArray(np.zeros([len(xx)-1,len(yy)-1]),dims=coordinate)
            #ds_tmp['center']=xr.DataArray(np.zeros([len(xx),len(yy),2]),dims=('x','y','center'))
    
            
                
            print('Box_size :',size_box[bb])
            for i in trange(len(xx)-1):
                for j in range(len(yy)-1):
                    sub_img=self._obj[xx[i]:xx[i]+size_box[bb],yy[j]:yy[j]+size_box[bb]]
                    if cutoff_sphere:
                        sub_img=sub_img.imP.circle_filter(vfill=0)
                # might not be the best when circle filter is on but in practice should not be a problem        
                    if len(np.unique(sub_img))!=1: # don't when to include uniform image in the data set
                        ds_ell=sub_img.imP.inertia_tensor()
                        ds_tmp.a1[i,j]=ds_ell.a1
                        ds_tmp.a2[i,j]=ds_ell.a2
                        ds_tmp.e1[i,j,:]=ds_ell.e1
                        ds_tmp.e2[i,j,:]=ds_ell.e2
                        ds_tmp.angle[i,j]=np.arctan(ds_ell.e2[0]/ds_ell.e2[1])
                    else:
                        ds_tmp.a1[i,j]=np.nan
                        ds_tmp.a2[i,j]=np.nan
                        ds_tmp.e1[i,j,:]=np.nan
                        ds_tmp.e2[i,j,:]=np.nan
                        
                
            
            ds_tmp['anisotropy']=1-ds_tmp.a2/ds_tmp.a1
            
            ds_tmp.coords[coordinate[0]]=self._obj[coordinate[0]][xx[0:-1]]
            ds_tmp.coords[coordinate[1]]=self._obj[coordinate[1]][yy[0:-1]]
            
            all_dict.append(ds_tmp)
            
            
            
            
            
                        
        return all_dict
                
    def split_img(self,ss_box,pix_shift=0):
        '''
        Divide the image in sub-image of box size size_box³
        :param ss_box: size of the cubic box
        :type ss_box: np.int
        :param pix_shift: (0) paved surface without the excess surface and without overlap (np.int) number of pixel shift for overlapping windows
        :type option: int
        '''
        ss=np.shape(self._obj)
        
        if pix_shift==0:
            xx=np.arange(0, ss[0], ss_box)
            yy=np.arange(0, ss[1], ss_box)
        else:
            xx=np.arange(0,ss[0]-ss_box,pix_shift)
            yy=np.arange(0,ss[1]-ss_box,pix_shift)
    
        center_box=[]
        sub_img=[]
            
        sub_img=[]
        center_box=[]
        for i in range(len(xx)-1):
            for j in range(len(yy)-1):
                sub_img.append(self._obj[xx[i]:xx[i]+ss_box,yy[j]:yy[j]+ss_box])
                center_box.append(np.array([xx[i]+ss_box/2,yy[j]+ss_box/2]))

        return center_box,sub_img
