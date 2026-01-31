
import numpy as np
from typing import Tuple, Optional
import warnings

class TraitementImages:
    """Classe principale pour le traitement d'images"""

    # 1.Conversions
    
    @staticmethod
    def rgb_to_grayscale(img_bgr: np.ndarray) -> np.ndarray:
        """Convertit BGR en niveaux de gris (OpenCV -> BGR)"""
        B = img_bgr[:, :, 0].astype(np.float32)
        G = img_bgr[:, :, 1].astype(np.float32)
        R = img_bgr[:, :, 2].astype(np.float32)
        
        # Formule standard ITU-R BT.601 (adaptée pour BGR)
        gray = 0.114 * B + 0.587 * G + 0.299 * R
        gray = np.clip(gray, 0, 255)
        
        return gray.astype(np.uint8)
    
    @staticmethod
    def rgb_to_yuv(img_rgb: np.ndarray) -> np.ndarray:
        """Convertit RGB vers YUV"""
        R = img_rgb[:, :, 0].astype(np.float32)
        G = img_rgb[:, :, 1].astype(np.float32)
        B = img_rgb[:, :, 2].astype(np.float32)
        
        Y = 0.299 * R + 0.587 * G + 0.114 * B
        U = B - Y
        V = R - Y
        
        return np.stack([Y, U, V], axis=-1)
    
    @staticmethod
    def rgb_to_hsv(img_rgb: np.ndarray) -> np.ndarray:
        """Convertit RGB vers HSV"""
        img_norm = img_rgb.astype(np.float32) / 255.0
        r, g, b = img_norm[:, :, 0], img_norm[:, :, 1], img_norm[:, :, 2]
        
        cmax = np.max(img_norm, axis=2)
        cmin = np.min(img_norm, axis=2)
        delta = cmax - cmin
        
        # Initialisation
        h = np.zeros_like(cmax)
        s = np.zeros_like(cmax)
        v = cmax
        
        # Calcul de H (Teinte)
        mask = delta != 0
        r_mask = mask & (cmax == r)
        g_mask = mask & (cmax == g)
        b_mask = mask & (cmax == b)
        
        h[r_mask] = 60 * (((g[r_mask] - b[r_mask]) / delta[r_mask]) % 6)
        h[g_mask] = 60 * (((b[g_mask] - r[g_mask]) / delta[g_mask]) + 2)
        h[b_mask] = 60 * (((r[b_mask] - g[b_mask]) / delta[b_mask]) + 4)
        
        # Calcul de S (Saturation)
        s[mask] = delta[mask] / cmax[mask]
        
        return np.stack([h, s, v], axis=-1)
    

    # 2.Seuillage
    
    @staticmethod
    def seuillage_fixe(img_gray: np.ndarray, threshold: int = 128) -> np.ndarray:
        """Seuillage binaire global"""
        return ((img_gray > threshold) * 255).astype(np.uint8)
    
    @staticmethod
    def seuillage_adaptatif(img_gray: np.ndarray, 
                           block_size: int = 11, 
                           C: int = 2) -> np.ndarray:
        """Seuillage adaptatif par moyenne locale"""
        if block_size % 2 == 0:
            raise ValueError("block_size doit être impair")
        
        h, w = img_gray.shape
        k = block_size // 2
        img_binary = np.zeros_like(img_gray)
        
        # Padding
        img_padded = np.pad(img_gray, k, mode='edge')
        
        for i in range(h):
            for j in range(w):
                neighborhood = img_padded[i:i+block_size, j:j+block_size]
                T_local = np.mean(neighborhood) - C
                img_binary[i, j] = 255 if img_gray[i, j] > T_local else 0
        
        return img_binary
    
    @staticmethod
    def seuillage_otsu(img_gray: np.ndarray) -> Tuple[np.ndarray, int]:
        """Seuillage d'Otsu - méthode automatique"""
        # Histogramme
        hist, _ = np.histogram(img_gray.flatten(), bins=256, range=(0, 256))
        prob = hist / img_gray.size
        
        best_threshold = 0
        max_variance = 0
        
        # Tester tous les seuils
        for T in range(256):
            w0 = prob[:T+1].sum()
            w1 = prob[T+1:].sum()
            
            if w0 == 0 or w1 == 0:
                continue
            
            mu0 = np.sum(np.arange(T+1) * prob[:T+1]) / w0
            mu1 = np.sum(np.arange(T+1, 256) * prob[T+1:]) / w1
            
            variance = w0 * w1 * (mu0 - mu1) ** 2
            
            if variance > max_variance:
                max_variance = variance
                best_threshold = T
        
        # Appliquer le seuil
        img_binary = TraitementImages.seuillage_fixe(img_gray, best_threshold)
        
        return img_binary, best_threshold
    

    # 3.Filtrage
    
    @staticmethod
    def convolution_2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Convolution 2D manuelle"""
        h_img, w_img = img.shape
        h_k, w_k = kernel.shape
        
        if h_k != w_k or h_k % 2 == 0:
            raise ValueError("Noyau carré impair requis")
        
        k = h_k // 2
        output = np.zeros((h_img - 2*k, w_img - 2*k), dtype=np.float32)
        img_float = img.astype(np.float32)
        
        for i in range(k, h_img - k):
            for j in range(k, w_img - k):
                region = img_float[i-k:i+k+1, j-k:j+k+1]
                output[i-k, j-k] = np.sum(region * kernel)
        
        output = np.clip(output, 0, 255)
        return output.astype(np.uint8)
    
    @staticmethod
    def filtre_gaussien(img: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """Filtre gaussien"""
        size = int(6 * sigma) + 1
        if size % 2 == 0:
            size += 1
        
        k = size // 2
        
        # Créer le noyau gaussien
        x = np.arange(-k, k+1)
        y = np.arange(-k, k+1)
        X, Y = np.meshgrid(x, y)
        
        kernel = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
        kernel /= kernel.sum()
        
        return TraitementImages.convolution_2d(img, kernel)
    
    @staticmethod
    def filtre_median(img: np.ndarray, size: int = 3) -> np.ndarray:
        """Filtre médian"""
        if size % 2 == 0:
            raise ValueError("Taille impaire requise")
        
        h, w = img.shape
        k = size // 2
        output = np.zeros_like(img)
        
        # Padding
        img_padded = np.pad(img, k, mode='edge')
        
        for i in range(k, h + k):
            for j in range(k, w + k):
                region = img_padded[i-k:i+k+1, j-k:j+k+1]
                output[i-k, j-k] = np.median(region)
        
        return output.astype(np.uint8)
    
    @staticmethod
    def filtre_sobel(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Filtre de Sobel"""
        kernel_x = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]])
        
        kernel_y = np.array([[-1, -2, -1],
                            [ 0,  0,  0],
                            [ 1,  2,  1]])
        
        grad_x = TraitementImages.convolution_2d(img, kernel_x)
        grad_y = TraitementImages.convolution_2d(img, kernel_y)
        
        magnitude = np.sqrt(grad_x.astype(np.float32)**2 + 
                           grad_y.astype(np.float32)**2)
        magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)
        
        return grad_x, grad_y, magnitude
    

    # 4.Morphologie
    
    @staticmethod
    def erosion(img_bin: np.ndarray, kernel: Optional[np.ndarray] = None) -> np.ndarray:
        """Érosion morphologique"""
        if kernel is None:
            kernel = np.ones((3, 3), dtype=np.uint8)
        
        h, w = img_bin.shape
        k_h, k_w = kernel.shape
        half_h, half_w = k_h // 2, k_w // 2
        
        output = np.zeros_like(img_bin)
        
        for i in range(half_h, h - half_h):
            for j in range(half_w, w - half_w):
                erosion_ok = True
                
                for ki in range(k_h):
                    for kj in range(k_w):
                        if kernel[ki, kj] == 1:
                            if img_bin[i+ki-half_h, j+kj-half_w] == 0:
                                erosion_ok = False
                                break
                    if not erosion_ok:
                        break
                
                if erosion_ok:
                    output[i, j] = 255
        
        return output
    
    @staticmethod
    def dilatation(img_bin: np.ndarray, kernel: Optional[np.ndarray] = None) -> np.ndarray:
        """Dilatation morphologique"""
        if kernel is None:
            kernel = np.ones((3, 3), dtype=np.uint8)
        
        h, w = img_bin.shape
        k_h, k_w = kernel.shape
        half_h, half_w = k_h // 2, k_w // 2
        
        output = np.zeros_like(img_bin)
        
        for i in range(h):
            for j in range(w):
                if img_bin[i, j] == 255:
                    for ki in range(k_h):
                        for kj in range(k_w):
                            if kernel[ki, kj] == 1:
                                ni = i + ki - half_h
                                nj = j + kj - half_w
                                if 0 <= ni < h and 0 <= nj < w:
                                    output[ni, nj] = 255
        
        return output
    
    @staticmethod
    def ouverture(img_bin: np.ndarray, kernel: Optional[np.ndarray] = None) -> np.ndarray:
        """Ouverture = Érosion suivie de Dilatation"""
        eroded = TraitementImages.erosion(img_bin, kernel)
        return TraitementImages.dilatation(eroded, kernel)
    
    @staticmethod
    def fermeture(img_bin: np.ndarray, kernel: Optional[np.ndarray] = None) -> np.ndarray:
        """Fermeture = Dilatation suivie d'Érosion"""
        dilated = TraitementImages.dilatation(img_bin, kernel)
        return TraitementImages.erosion(dilated, kernel)
    
    
    # 5.Utilitaires
    
    @staticmethod
    def calculer_histogramme(img: np.ndarray) -> np.ndarray:
        """Calcule l'histogramme d'une image"""
        hist = np.zeros(256, dtype=np.int32)
        
        if len(img.shape) == 2:  # Niveaux de gris
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    hist[img[i, j]] += 1
        else:  # Couleur
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    valeur = int(np.mean(img[i, j]))
                    hist[valeur] += 1
        
        return hist
    
    @staticmethod
    def redimensionner(img: np.ndarray, new_width: int, new_height: int) -> np.ndarray:
        """Redimensionne une image (plus proche voisin)"""
        h_orig, w_orig = img.shape[:2]
        h_ratio = h_orig / new_height
        w_ratio = w_orig / new_width
        
        if len(img.shape) == 3:
            output = np.zeros((new_height, new_width, img.shape[2]), dtype=img.dtype)
        else:
            output = np.zeros((new_height, new_width), dtype=img.dtype)
        
        for i in range(new_height):
            for j in range(new_width):
                i_orig = int(i * h_ratio)
                j_orig = int(j * w_ratio)
                output[i, j] = img[i_orig, j_orig]
        
        return output