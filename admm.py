import numpy as np
import scipy.ndimage
import skimage.io
import matplotlib.pyplot as plt
import scipy.fftpack
import sys
ros_path = "/opt/ros/kinetic/lib/python2.7/dist-packages"

if ros_path in sys.path:
    sys.path.remove(ros_path)
import cv2
import scipy.ndimage
import numpy as np


def ForwardD(U, beta):
    Dux = beta[0]*np.hstack((np.diff(U,1,1),(U[:,0] - U[:,-1]).reshape(-1,1)))
    
    Duy = beta[1]*np.vstack((np.diff(U,1,0), (U[0,:] - U[-1,:])))

    Duz = np.zeros_like(U)

    return Dux, Duy, Duz


def Dive(X,Y,Z, beta):
    DtXYZ = np.hstack(((X[:,-1] - X[:,0]).reshape(-1,1),-np.diff(X,1,1)))
    

    DtXYZ = beta[0] * DtXYZ + beta[1]*np.vstack((Y[-1,:] - Y[0,:],
                                                 -np.diff(Y,1,0)))

    return DtXYZ    


def deconvtvl2(image,H,mu):
    
    # % out = deconvtvl1(g, H, mu, opts)
    # % deconvolves image g by solving the following TV minimization problem
    # %
    # % min (mu/2) || Hf - g ||_1 + ||f||_TV
    # %
    # % where ||f||_TV = sqrt( a||Dxf||^2 + b||Dyf||^2 c||Dtf||^2),
    # % Dxf = f(x+1,y, t) - f(x,y,t)
    # % Dyf = f(x,y+1, t) - f(x,y,t)
    # % Dtf = f(x,y, t+1) - f(x,y,t)
    # %
    # % Input:      image      - the observed image, can be gray scale, or color
    # %             H      - point spread function
    # %            mu      - regularization parameter
    # %     opts.rho_r     - initial penalty parameter for ||u-Df||   {2}
    # %     opts.rho_o     - initial penalty parameter for ||Hf-g-r|| {50}
    # %     opts.beta      - regularization parameter [a b c] for weighted TV norm {[1 1 2.5]}
    # %     opts.gamma     - update constant for rho_r {2}
    # %     opts.max_itr   - maximum iteration {20}
    # %     opts.alpha     - constant that determines constraint violation {0.7}
    # %     opts.tol       - tolerance level on relative change {1e-3}
    # %     opts.print     - print screen option {false}
    # %     opts.f         - initial f  {g}
    # %     opts.y1        - initial y1 {0}
    # %     opts.y2        - initial y2 {0}
    # %     opts.y3        - initial y3 {0}
    # %     opts.z         - initial z {0}
    # %     ** default values of opts are given in { }.
    # %
    # % Output: out.f      - output video
    # %         out.itr    - total number of iterations elapsed
    # %         out.relchg - final relative change
    # %         out.Df1    - Dxf, f is the output video
    # %         out.Df2    - Dyf, f is the output video
    # %         out.Df3    - Dtf, f is the output video
    # %         out.y1     - Lagrange multiplier for Df1
    # %         out.y2     - Lagrange multiplier for Df2
    # %         out.y3     - Lagrange multiplier for Df3
    # %         out.rho_r  - final penalty parameter
    # %
    # % Stanley Chan
    # % Copyright 2010
    # % University of California, San Diego
    # %
    # % Last Modified:
    # % 30 Apr, 2010 (deconvtv)
    # %  4 May, 2010 (deconvtv)
    # %  5 May, 2010 (deconvtv)
    # %  4 Aug, 2010 (deconvtv_L1)
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    rows,cols = image.shape
    
    # rho_o = 50
    rho = 2
    gamma = 2
    max_itr = 20
    tol = 1e-3
    alpha = 0.7
    f = image
    
    y1 = np.zeros_like(image)
    y2 = np.zeros_like(image)
    y3 = np.zeros_like(image)
    
    u1 = np.zeros_like(image)
    u2 = np.zeros_like(image)
    u3 = np.zeros_like(image)
    
    
    
    z = np.zeros_like(image)
    beta = np.array([1,1,0])
    eigHtH = np.ones_like(image) #np.abs(np.fft.fftn(H,(rows,cols,1)))**2
    eigDtD = np.abs(beta[0]*np.fft.fftn(np.array([[1,-1]]),(rows,cols)))**2 + np.abs(beta[1]*np.fft.fftn(np.array([[1],[-1]]),(rows,cols)))**2
    
    eigEtE = 0
    
    # Htg = cv2.filter2D(image, -1, H)
    # Htg = scipy.ndimage.correlate(image, H, mode='wrap')
    
    Htg = image   ############################

    
    Df1,Df2,Df3 = ForwardD(f.copy(),beta)

    w = np.zeros_like(image)
    rnorm = np.sqrt(np.linalg.norm(Df1.flatten())**2 + np.linalg.norm(Df2.flatten())**2 + np.linalg.norm(Df3.flatten())**2)
    
    
    for itr in range(max_itr):
        f_old = f
        
        rhs   = np.fft.fftn((mu/rho)*Htg + Dive(u1-(1/rho)*y1,  u2-(1/rho)*y2, u3-(1/rho)*y3,beta));
        eigA = (mu/rho)*eigHtH + eigDtD + eigEtE
        f = np.real(np.fft.ifftn(rhs.copy())/eigA.copy())
        
        Df1, Df2, Df3 = ForwardD(f,beta)
        
        v1 = Df1 + (1/rho) * y1
        
        v2 = Df2 + (1/rho) * y2
        
        v3 = Df3 + (1/rho) * y3
        
        
        v = np.sqrt(v1**2 + v2**2 + v3**2)
        v[v==0] = 1
        
        v = np.maximum(v - 1/rho, np.zeros_like(v)) / v
        
        
        u1 = v1 * v
        u2 = v2 * v
        u3 = v3 * v
        
        y1 = y1 - rho*(u1 - Df1)
        
        y2 = y2 - rho*(u2 - Df2)
        y3 = y3 - rho*(u3 - Df3)
        
        rnorm_old = rnorm
        rnorm = np.sqrt(np.linalg.norm(Df1.flatten()-u1.flatten())**2 + np.linalg.norm(Df2.flatten() - u2.flatten())**2 + np.linalg.norm(Df3.flatten() - u3.flatten())**2)
        
        if rnorm > alpha*rnorm_old:
            rho = rho * gamma
            
        relchg = np.linalg.norm(f.flatten() - f_old.flatten()) / np.linalg.norm(f_old.flatten())
        if relchg < tol:
            break
        
    return f




def deconvtvl1(image,H,mu):
    
    # % out = deconvtvl1(g, H, mu, opts)
    # % deconvolves image g by solving the following TV minimization problem
    # %
    # % min (mu/2) || Hf - g ||_1 + ||f||_TV
    # %
    # % where ||f||_TV = sqrt( a||Dxf||^2 + b||Dyf||^2 c||Dtf||^2),
    # % Dxf = f(x+1,y, t) - f(x,y,t)
    # % Dyf = f(x,y+1, t) - f(x,y,t)
    # % Dtf = f(x,y, t+1) - f(x,y,t)
    # %
    # % Input:      image      - the observed image, can be gray scale, or color
    # %             H      - point spread function
    # %            mu      - regularization parameter
    # %     opts.rho_r     - initial penalty parameter for ||u-Df||   {2}
    # %     opts.rho_o     - initial penalty parameter for ||Hf-g-r|| {50}
    # %     opts.beta      - regularization parameter [a b c] for weighted TV norm {[1 1 2.5]}
    # %     opts.gamma     - update constant for rho_r {2}
    # %     opts.max_itr   - maximum iteration {20}
    # %     opts.alpha     - constant that determines constraint violation {0.7}
    # %     opts.tol       - tolerance level on relative change {1e-3}
    # %     opts.print     - print screen option {false}
    # %     opts.f         - initial f  {g}
    # %     opts.y1        - initial y1 {0}
    # %     opts.y2        - initial y2 {0}
    # %     opts.y3        - initial y3 {0}
    # %     opts.z         - initial z {0}
    # %     ** default values of opts are given in { }.
    # %
    # % Output: out.f      - output video
    # %         out.itr    - total number of iterations elapsed
    # %         out.relchg - final relative change
    # %         out.Df1    - Dxf, f is the output video
    # %         out.Df2    - Dyf, f is the output video
    # %         out.Df3    - Dtf, f is the output video
    # %         out.y1     - Lagrange multiplier for Df1
    # %         out.y2     - Lagrange multiplier for Df2
    # %         out.y3     - Lagrange multiplier for Df3
    # %         out.rho_r  - final penalty parameter
    # %
    # % Stanley Chan
    # % Copyright 2010
    # % University of California, San Diego
    # %
    # % Last Modified:
    # % 30 Apr, 2010 (deconvtv)
    # %  4 May, 2010 (deconvtv)
    # %  5 May, 2010 (deconvtv)
    # %  4 Aug, 2010 (deconvtv_L1)
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    rows,cols = image.shape
    
    rho_o = 50
    rho_r = 2
    gamma = 2
    max_itr = 20
    tol = 1e-3
    alpha = 0.7
    f = image
    
    y1 = np.zeros_like(image)
    y2 = np.zeros_like(image)
    y3 = np.zeros_like(image)
    z = np.zeros_like(image)
    beta = np.array([1,1,0])
    eigHtH = np.ones_like(image) #np.abs(np.fft.fftn(H,(rows,cols,1)))**2
    eigDtD = np.abs(beta[0]*np.fft.fftn(np.array([[1,-1]]),(rows,cols)))**2 + np.abs(beta[1]*np.fft.fftn(np.array([[1],[-1]]),(rows,cols)))**2
    
    eigEtE = 0
    
    # Htg = cv2.filter2D(image, -1, H)
    # Htg = scipy.ndimage.correlate(image, H, mode='wrap')
    
    Htg = image   ############################

    
    Df1,Df2,Df3 = ForwardD(f.copy(),beta)

    w = np.zeros_like(image)
    rnorm = np.sqrt(np.linalg.norm(Df1.flatten())**2 + np.linalg.norm(Df2.flatten())**2 + np.linalg.norm(Df3.flatten())**2)
    
    
    for itr in range(max_itr):
        
        v1 = Df1 + (1/rho_r) * y1
        
        v2 = Df2 + (1/rho_r) * y2
        
        v3 = Df3 + (1/rho_r) * y3
        
        
        v = np.sqrt(v1**2 + v2**2 + v3**2)
        v[v==0] = 1e-6
        
        v = np.maximum(v - 1/rho_r, np.zeros_like(v)) / v
        
        
        u1 = v1 * v
        u2 = v2 * v
        u3 = v3 * v
        
        r = np.maximum(np.abs(w + 1/rho_o * z) - mu/rho_o,np.zeros_like(w)) * np.sign(w+1/rho_o*z)
        
        f_old = f
        rhs = rho_o*Htg + rho_o*r-z + Dive(rho_r*u1-y1,rho_r*u2-y2,rho_r*u3-y3,beta)
        
        eigA = rho_o * eigHtH + rho_r*eigDtD + rho_r*eigEtE
        
        f = np.real(np.fft.ifftn(np.fft.fftn(rhs.copy())/eigA.copy()))
             
        Df1, Df2, Df3 = ForwardD(f,beta)

        w = f-image
        
        y1 = y1 - rho_r*(u1 - Df1)
        
        y2 = y2 - rho_r*(u2 - Df2)
        y3 = y3 - rho_r*(u3 - Df3)

        z = z - rho_o*(r-w)
        
        rnorm_old = rnorm
        rnorm = np.sqrt(np.linalg.norm(Df1.flatten()-u1.flatten())**2 + np.linalg.norm(Df2.flatten() - u2.flatten())**2 + np.linalg.norm(Df3.flatten() - u3.flatten())**2)
        
        if rnorm > alpha*rnorm_old:
            rho_r = rho_r * gamma
        
        relchg = np.linalg.norm(f.flatten() - f_old.flatten()) / np.linalg.norm(f_old.flatten())
        
        if relchg < tol:
            break
        
    return f
    
def PlugPlayADMM_deblur(blurry_image,kernel,norm='l1',lam=0.01,rho=1,gamma=1,max_itr=20):
    '''
    Parameters
    ----------
    blurry_image : image
    kernel : blur kernel
    lam : TYPE
        regularization constant
    opts : list of params: rho, gamma, max_itrs

    Returns
    -------
    deblurred img

    '''
    
    y = blurry_image
    h = kernel
    # rho,gamma,max_itr = opts
    
    tol = 1e-4
    
    dim = blurry_image.shape
    N = blurry_image.shape[0] * blurry_image.shape[1]
    
    Hty = scipy.ndimage.correlate(y, h, mode='reflect')

    eigHtH = np.abs(np.fft.fftn(h,dim))**2

    v = np.ones(dim) * 0.5
    x = v
    u = np.zeros(dim)
    residual = 1000000
    
    itr = 1
    while (residual > tol and itr <= max_itr):
        x_old = x
        v_old = v
        u_old = u
        
        xtilde = v-u
        
        rhs = np.fft.fftn(Hty + rho*xtilde,dim)
        x = np.real(np.fft.ifftn(rhs/(eigHtH+rho)))

        vtilde = x+u
        
        bound = [0,1]
        upper = np.ones_like(vtilde) * bound[1]
        lower = np.ones_like(vtilde) * bound[0]
        
        vtilde = np.minimum(np.maximum(vtilde,lower),upper)
        
        sigma = np.sqrt(lam/rho)
        
        if norm == 'l2':
            v = deconvtvl2(vtilde,1,1/sigma**2)
        elif norm == 'l1':
            v = deconvtvl1(vtilde,1,1/sigma**2)
        
        
        u = u + (x-v)
        
        rho = rho*gamma
        
        residualx = (1/np.sqrt(N))*np.sqrt(np.sum(x-x_old)**2)
        residualv = (1/np.sqrt(N))*np.sqrt(np.sum(v-v_old)**2)
        residualu = (1/np.sqrt(N))*np.sqrt(np.sum(u-u_old)**2)
        
        print(residualx,'\t',residualv,'\t',residualu)
        
        residual = residualx + residualv + residualu
        itr+=1
    
    return v

def get_psnr(good,bad):
    result = -10*np.log10(np.mean((good.flatten() - bad.flatten())**2))
    return result

if __name__ == "__main__":
    filename = "House256.png"
    img = skimage.img_as_float32(skimage.io.imread(filename))
    plt.imshow(img,cmap='gray')
    plt.title("Original")
    plt.xticks([]), plt.yticks([])
    plt.show()
    
    plt.imshow(img[120:175,100:150],cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.show()
    
    
    h = np.ones(9) / 9
    h = h.reshape(1,9)
    
    y = scipy.ndimage.correlate(img, h, mode='wrap')
    
    plt.imshow(y,cmap='gray')
    plt.title("Blurred with horizontal kernel")
    plt.xticks([]), plt.yticks([])
    plt.show()
    
    plt.imshow(y[120:175,100:150],cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.show()
    
    lam = 0.01
    rho = 1
    gamma = 1
    max_iters = 20
    
    # out = PlugPlayADMM_deblur(y,h,lam,(rho,gamma,max_iters))
    
    out = PlugPlayADMM_deblur(y,h,'l2',lam,rho,gamma,max_iters)
    psnr = get_psnr(out,img)
    
    psnr_str = "%0.2f" % psnr
    plt.imshow(out,cmap='gray')
    plt.title("Deblur result with ADMM (PSNR = " + psnr_str + ")")
    plt.xticks([]), plt.yticks([])
    plt.show()
    
    plt.imshow(out[120:175,100:150],cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.show()
    
    
    
    print("PSNR:", psnr)
    