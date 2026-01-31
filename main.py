
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys


sys.path.append('yehia-elkh/OpenCv/vision-artificielle-traitement-images/')

from src.traitement_images import TraitementImages

class ImageDemo3:
    """Démonstration avec 3 images spécifiques"""
    
    def __init__(self):
        self.processor = TraitementImages()
        
        # Chemins des images
        self.image_paths = {
            'cr7_messi1': Path('images/cr7-messi.jpg'),
            'cr7_messi2': Path('images/cr7-messi2.jpg'),
            'opencv_logo': Path('images/opencv_logo.webp')
        }
        
        # Vérifier que les images existent
        self.check_images_exist()
    
    def check_images_exist(self):
        """Vérifie que les 3 images existent"""
        print("\n")
        print("Vérification des images\n")

        
        all_exist = True
        for name, path in self.image_paths.items():
            if path.exists():
                print(f"V {name}: {path.name} ({path.stat().st_size / 1024:.1f} KB)")
            else:
                print(f"X {name}: {path.name} - Fichier introuvable!")
                all_exist = False
        
        if not all_exist:
            print("\n! Certaines images sont manquantes!")
            print("   Veuillez placer les images dans le dossier 'images/'")
            sys.exit(1)
        
        print("\n\n")
    
    def load_and_analyze_image(self, image_path, image_name):
        """Charge et analyse une image"""
        print(f"\nAnalyse de: {image_name} ({image_path.name})")
        
        # Charger l'image
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            raise ValueError(f"Impossible de charger l'image: {image_path}")
        
        # Convertir en RGB pour l'affichage
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_gray = self.processor.rgb_to_grayscale(img_bgr)
        
        # Informations de base
        print(f"   Dimensions: {img_bgr.shape}")
        print(f"   Type: {img_bgr.dtype}")
        print(f"   Intensité moyenne: {np.mean(img_gray):.1f}")
        print(f"   Intensité min/max: {np.min(img_gray)}/{np.max(img_gray)}")
        
        return {
            'name': image_name,
            'filename': image_path.name,
            'bgr': img_bgr,
            'rgb': img_rgb,
            'gray': img_gray,
            'shape': img_bgr.shape
        }
    
    def apply_all_techniques(self, image_data):
        """Applique toutes les techniques à une image"""
        results = {}
        
        print(f"\nApplications sur: {image_data['name']}")
        
        # 1. Seuillages
        print("   1. Seuillages:")
        results['binary_fixed'] = self.processor.seuillage_fixe(image_data['gray'], 128)
        results['binary_adaptive'] = self.processor.seuillage_adaptatif(image_data['gray'], block_size=15, C=5)
        results['binary_otsu'], otsu_threshold = self.processor.seuillage_otsu(image_data['gray'])
        results['otsu_threshold'] = otsu_threshold
        
        print(f"      Seuil Otsu optimal: {otsu_threshold}")
        
        # 2. Filtrages
        print("   2. Filtrages:")
        results['gaussian'] = self.processor.filtre_gaussien(image_data['gray'], sigma=1.5)
        results['median'] = self.processor.filtre_median(image_data['gray'], size=3)
        results['sobel_x'], results['sobel_y'], results['sobel_mag'] = self.processor.filtre_sobel(image_data['gray'])
        
        # 3. Morphologie (sur l'image Otsu)
        print("   3. Morphologie:")
        kernel = np.ones((3, 3), dtype=np.uint8)
        results['eroded'] = self.processor.erosion(results['binary_otsu'], kernel)
        results['dilated'] = self.processor.dilatation(results['binary_otsu'], kernel)
        results['opened'] = self.processor.ouverture(results['binary_otsu'], kernel)
        results['closed'] = self.processor.fermeture(results['binary_otsu'], kernel)
        
        # 4. Conversions de couleur
        print("   4. Conversions de couleur:")
        results['yuv'] = self.processor.rgb_to_yuv(image_data['rgb'])
        results['hsv'] = self.processor.rgb_to_hsv(image_data['rgb'])
        
        # 5. Histogramme
        print("   5. Calcul d'histogramme:")
        results['histogram'] = self.processor.calculer_histogramme(image_data['gray'])
        
        return results
    
    def create_comparison_plot(self, image1_data, image2_data, image3_data):
        """Crée un graphique comparatif des 3 images"""
        fig, axes = plt.subplots(3, 5, figsize=(18, 10))
        
        images_info = [
            (image1_data, "CR7 & Messi 1"),
            (image2_data, "CR7 & Messi 2"),
            (image3_data, "Logo OpenCV")
        ]
        
        for row, (img_data, title) in enumerate(images_info):
            # Original RGB
            axes[row, 0].imshow(img_data['rgb'])
            axes[row, 0].set_title(f"{title}\nOriginal RGB", fontsize=10)
            axes[row, 0].axis('off')
            
            # Niveaux de gris
            axes[row, 1].imshow(img_data['gray'], cmap='gray')
            axes[row, 1].set_title('Niveaux de Gris', fontsize=10)
            axes[row, 1].axis('off')
            
            # Seuillage Otsu
            results = self.apply_all_techniques(img_data)
            axes[row, 2].imshow(results['binary_otsu'], cmap='gray')
            axes[row, 2].set_title(f'Otsu (T={results["otsu_threshold"]})', fontsize=10)
            axes[row, 2].axis('off')
            
            # Filtre Gaussien
            axes[row, 3].imshow(results['gaussian'], cmap='gray')
            axes[row, 3].set_title('Gaussien G=1.5', fontsize=10)
            axes[row, 3].axis('off')
            
            # Sobel Magnitude
            axes[row, 4].imshow(results['sobel_mag'], cmap='gray')
            axes[row, 4].set_title('Sobel (Magnitude)', fontsize=10)
            axes[row, 4].axis('off')
        
        plt.suptitle('Comparaison des 3 Images avec Traitements', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig('comparison_3_images.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return images_info
    
    def create_detailed_plots(self, image_data, results, image_num):
        """Crée des graphiques détaillés pour chaque image"""
        # Plot 1: Original vs Traitements de base
        fig1, axes1 = plt.subplots(2, 4, figsize=(16, 8))
        axes1 = axes1.ravel()
        
        treatments = [
            ('Original RGB', image_data['rgb'], False),
            ('Niveaux de Gris', image_data['gray'], True),
            (f'Otsu (T={results["otsu_threshold"]})', results['binary_otsu'], True),
            ('Adaptatif', results['binary_adaptive'], True),
            ('Gaussien', results['gaussian'], True),
            ('Médian', results['median'], True),
            ('Sobel', results['sobel_mag'], True),
            ('YUV (Luminance)', results['yuv'][:, :, 0], True),
        ]
        
        for idx, (title, img, is_gray) in enumerate(treatments):
            if is_gray:
                axes1[idx].imshow(img, cmap='gray')
            else:
                axes1[idx].imshow(img)
            axes1[idx].set_title(title, fontsize=9)
            axes1[idx].axis('off')
        
        plt.suptitle(f'Traitements de base - {image_data["name"]}', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'treatments_image{image_num}.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Plot 2: Morphologie
        fig2, axes2 = plt.subplots(2, 3, figsize=(12, 8))
        axes2 = axes2.ravel()
        
        morph_treatments = [
            ('Image Otsu', results['binary_otsu']),
            ('Érosion', results['eroded']),
            ('Dilatation', results['dilated']),
            ('Ouverture', results['opened']),
            ('Fermeture', results['closed']),
            ('Différence', cv2.absdiff(results['dilated'], results['eroded'])),
        ]
        
        for idx, (title, img) in enumerate(morph_treatments):
            axes2[idx].imshow(img, cmap='gray')
            axes2[idx].set_title(title, fontsize=10)
            axes2[idx].axis('off')
        
        plt.suptitle(f'Morphologie Mathématique - {image_data["name"]}', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'morphology_image{image_num}.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Plot 3: Histogramme
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(image_data['gray'], cmap='gray')
        plt.title(f'{image_data["name"]} - Niveaux de Gris')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.bar(range(256), results['histogram'], width=1.0, color='blue', alpha=0.7)
        plt.axvline(x=results['otsu_threshold'], color='red', linestyle='--', 
                   label=f'Seuil Otsu: {results["otsu_threshold"]}')
        plt.title('Histogramme')
        plt.xlabel('Intensité')
        plt.ylabel('Fréquence')
        plt.xlim([0, 255])
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'histogram_image{image_num}.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def create_final_summary(self, all_results):
        """Crée un résumé final avec statistiques"""
        print("\n\n")
        print("Résumé Statistique des 3 images")
        print("\n\n")
        
        summary_data = []
        
        for i, (img_data, results) in enumerate(all_results, 1):
            stats = {
                'Image': img_data['name'],
                'Fichier': img_data['filename'],
                'Dimensions': f"{img_data['shape'][1]}×{img_data['shape'][0]}",
                'Canaux': img_data['shape'][2] if len(img_data['shape']) == 3 else 1,
                'Moyenne Gris': f"{np.mean(img_data['gray']):.1f}",
                'Écart-type': f"{np.std(img_data['gray']):.1f}",
                'Seuil Otsu': results['otsu_threshold'],
                'Contraste': f"{(np.max(img_data['gray']) - np.min(img_data['gray'])) / 255 * 100:.1f}%"
            }
            summary_data.append(stats)
            
            print(f"\n {img_data['name']}:")
            for key, value in stats.items():
                print(f"   {key}: {value}")
        
        # Créer un tableau visuel
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.axis('tight')
        ax.axis('off')
        
        # Données du tableau
        columns = ['Image', 'Dimensions', 'Moyenne Gris', 'Seuil Otsu', 'Contraste']
        table_data = [[d[col] for col in columns] for d in summary_data]
        
        # Créer le tableau
        table = ax.table(cellText=table_data, colLabels=columns, 
                        cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        plt.title('Comparaison Statistique des 3 Images', fontsize=14, pad=20)
        plt.savefig('statistical_summary.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Graphique de comparaison des seuils Otsu
        fig, ax = plt.subplots(figsize=(8, 5))
        images = [d['Image'] for d in summary_data]
        otsu_values = [d['Seuil Otsu'] for d in summary_data]
        
        bars = ax.bar(images, otsu_values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax.set_ylabel('Seuil Otsu', fontsize=12)
        ax.set_title('Comparaison des Seuils Otsu', fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Ajouter les valeurs sur les barres
        for bar, value in zip(bars, otsu_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                   f'{value}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('otsu_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def run(self):
        """Exécute la démonstration complète"""
        print("\n\n")
        print("Démonstration complète - 3 Images")
        print("\n\n")
        
        # 1. Charger les 3 images
        print("Chargement des images:")
        image1 = self.load_and_analyze_image(self.image_paths['cr7_messi1'], "CR7 & Messi 1")
        image2 = self.load_and_analyze_image(self.image_paths['cr7_messi2'], "CR7 & Messi 2")
        image3 = self.load_and_analyze_image(self.image_paths['opencv_logo'], "Logo OpenCV")
        
        # 2. Créer la comparaison
        print("\n Créer la comparaison:")
        all_images = self.create_comparison_plot(image1, image2, image3)
        
        # 3. Traitements détaillés pour chaque image
        print("\n Traitements:")
        all_results = []
        
        for i, (img_data, title) in enumerate(all_images, 1):
            print(f"\n   Image {i}: {title}")
            results = self.apply_all_techniques(img_data)
            all_results.append((img_data, results))
            self.create_detailed_plots(img_data, results, i)
        
        # 4. Résumé final
        print("\n Création Du Résumé final:")
        self.create_final_summary(all_results)
        
        # 5. Message de fin
        print("\n\n\n")
        print(" Démonstration Terminée avec succés!")
        print("\n")
        print("\nFichiers générés:")
        print("   - comparison_3_images.png")
        print("   - treatments_image1.png, treatments_image2.png, treatments_image3.png")
        print("   - morphology_image1.png, morphology_image2.png, morphology_image3.png")
        print("   - histogram_image1.png, histogram_image2.png, histogram_image3.png")
        print("   - statistical_summary.png")
        print("   - otsu_comparison.png")
        print("\nStatistiques sauvegardées dans les graphiques.")
        print("\n\n\n")

def main():
    """Point d'entrée principal"""
    demo = ImageDemo3()
    demo.run()

if __name__ == "__main__":
    main()