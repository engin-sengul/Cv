import cv2 #Bu kütüphane görüntü işleme işlevlerini içerir.
from cvzone.HandTrackingModule import HandDetector # (HandTrackingModule)Bu modül, elleri tespit etmek ve izlemek için kullanılan işlevleri sağlar.
import numpy as np # NumPy, Python'da bilimsel hesaplamalar ve çok boyutlu diziler için kullanılan bir kütüphanedir.
import math # Bu kütüphane matematiksel işlemler yapmak için kullanılır.
import time # Bu kütüphane, zamanla ilgili işlemler yapmak için kullanılır, örneğin gecikme süresi eklemek veya zamanı ölçmek.


cap =cv2.VideoCapture(0) # Bu, video akışını yakalamak için bir VideoCapture nesnesi oluşturur.
detector = HandDetector(maxHands=1) # Bu nesne, elleri tespit etmek ve izlemek için kullanılan işlevleri sağlar.
offset=20 # elin etrafında çizilecek dikdörtgenin kenarlarının genişliğini belirler.
imgSize=300 # eli kırpıp yeniden boyutlandırmak için kullanılacak olan hedef görüntü boyutunu belirler.

folder = "Data/E" # el hareketlerinin kaydedileceği klasörün yolunu belirtir.
counter =0 # Toplanan veri sayısını takip etmek için kullanılır. Başlangıçta sıfır olarak ayarlanır.

while True:
    success,img = cap.read() #success değişkeni, karenin başarıyla okunup okunmadığını belirtir. Okunan kare, img değişkenine atanır.
    hands,img= detector.findHands(img) #Bu işlem, görüntüdeki elleri algılar ve tespit edilen elleri içeren bir liste olan hands değişkenine ve işlenmiş görüntüye (img) geri döner.
    if hands:  # Görüntüde bir el var ise aşağıdaki kod bloğunu çalıştırır.
        hand = hands[0] # İlk tespit edilen eli seçer ve hand değişkenine atar.
        x,y,w,h=hand['bbox'] # x ve y, sınırlayıcı dikdörtgenin sol üst köşesinin koordinatlarını, w ve h ise genişlik ve yüksekliği temsil eder. Bu bilgiler, elin konumunu ve boyutunu belirler.
        
        
        imgWhite=np.ones((imgSize,imgSize,3),np.uint8)*255  #Bu satırda, imgWhite adında bir matris oluşturulur ve boyutu imgSizeximgSizex3 olarak ayarlanır. 3=RGB
        

        imgCrop= img[y-offset:y+offset+h,x-offset:x+offset+w]  # Bu satır, orjinal görüntüden elin bölgesini seçer ve bu bölgeyi imgCrop değişkenine atar. 
        
        imgCropShape = imgCrop.shape  # Bu satırda, imgCrop alt görüntüsünün boyutlarını elde etmek için shape özelliği kullanılır. 
        
        imgWhite[0:imgCropShape[0],0:imgCropShape[1]]=imgCrop #Bu satırda, imgCrop alt görüntüsü, imgWhite matrisinin belirli bir bölgesine atılır. 
        #Bu işlem, imgCrop alt görüntüsünü, imgWhite matrisindeki belirli bir bölgeye yerleştirir. 
        
       
        
        aspectRatio=h/w # Bu satır, elin sınırlayıcı dikdörtgeninin yükseklik ve genişlik değerlerini kullanarak aspectRatio değişkenini hesaplar.
        try:
            if aspectRatio>1:
                k=imgSize/h  # Yükseklik skalasını belirlemek için imgSize (hedef yükseklik) değerini el sınırlayıcı dikdörtgeninin yüksekliği (h) ile bölerek k değerini hesaplar.
                wCal=math.ceil(k*w) # Genişlik skalasını belirlemek için k değerini el sınırlayıcı dikdörtgeninin genişliği (w) ile çarparak wCal değerini hesaplar.
                imgResize = cv2.resize(imgCrop,(wCal,imgSize)) # Alt görüntüsünü, wCal genişlik ve imgSize yükseklik değerlerine göre yeniden boyutlandırır.
                imgResizeShape = imgResize.shape # Yeniden boyutlandırılmış imgResize alt görüntüsünün boyutlarını elde etmek için shape özelliğini kullanır.
                wGap=math.ceil((imgSize-wCal)/2)    # Bu, genişlik açısından ortalamak için kullanılır.        
                imgWhite[:,wGap:wCal+wGap]=imgResize #imgResize alt görüntüsünü, imgWhite matrisindeki ilgili bölgeye yerleştirir.
            else:
                k=imgSize/w # Genişlik skalasını belirlemek için imgSize (hedef genişlik) değerini el sınırlayıcı dikdörtgeninin genişliği (w) ile bölerek k değerini hesaplar.
                hCal=math.ceil(k*h) # Yükseklik skalasını belirlemek için k değerini el sınırlayıcı dikdörtgeninin yüksekliği (h) ile çarparak hCal değerini hesaplar. 
                imgResize = cv2.resize(imgCrop,(imgSize,hCal)) # imgCrop alt görüntüsünü, imgSize genişlik ve hCal yükseklik değerlerine göre yeniden boyutlandırır
                imgResizeShape = imgResize.shape #Yeniden boyutlandırılmış imgResize alt görüntüsünün boyutlarını elde etmek için shape özelliğini kullanır.
                hGap=math.ceil((imgSize-hCal)/2)    # Bu, yükseklik açısından ortalamak için kullanılır.        
                imgWhite[hGap:hCal+hGap,:]=imgResize # imgResize alt görüntüsünü, imgWhite matrisindeki ilgili bölgeye yerleştirir.

            cv2.imshow("ImageCrop",imgCrop)
            cv2.imshow("ImageWhite",imgWhite)
            #Bu kod, imgCrop ve imgWhite'ı ayrı ayrı göstermek için iki farklı pencere oluşturur ve bu pencerelerde ilgili görüntüleri gösterir.
        except:
            print("Error: Aspect ratio is not satisfied.")
        
       
    cv2.imshow("Image",img)
    key = cv2.waitKey(1)
    if key==ord("s"):
        counter +=1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',imgWhite)
        print(counter)
        #Bu kısımda, img görüntüsünü "Image" adlı bir pencerede gösterir ve kullanıcının "s" tuşuna basıp basmadığını kontrol eder. 
        # Eğer kullanıcı "s" tuşuna basarsa, counter değerini artırır, imgWhite matrisini belirtilen klasöre kaydeder ve counter değerini ekrana yazdırır.
        # Bu işlem, her "s" tuşuna basıldığında imgWhite'ın kaydedilmesini ve counter değerinin güncellenmesini sağlar.
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    