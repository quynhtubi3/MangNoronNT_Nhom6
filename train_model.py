import os
import warnings
import cv2
import keras
import matplotlib.pyplot as plt
import matplotlib.style as style
import numpy as np
import pandas as pd
from PIL import Image
from keras import models, layers, optimizers
from keras.applications import VGG16
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Dropout, Flatten
from keras.models import Model
from keras.preprocessing import image as image_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


# Dinh nghia cac bien

gestures = {'L_': 'L',
           'fi': 'E',
           'ok': 'F',
           'pe': 'V',
           'pa': 'B'
            }

gestures_map = {'E': 0,
                'L': 1,
                'F': 2,
                'V': 3,
                'B': 4
                }


gesture_names = {0: 'E',
                 1: 'L',
                 2: 'F',
                 3: 'V',
                 4: 'B'}


image_path = 'data'
models_path = 'models/saved_model.hdf5'
rgb = False
imageSize = 224


# Ham xu ly anh resize ve 224x224 va chuyen ve numpy array
def process_image(path):
    img = Image.open(path) #lấy hình ảnh theo đường dẫn path
    img = img.resize((imageSize, imageSize)) # resize hình ảnh theo imageSize x imgSize, trong bài là 224
    img = np.array(img) # chuyển ảnh thành mảng numpy để đưa vào mô hình máy để học
    return img

#mảng numpy để đưa vào mô hình học máy

# Xu ly du lieu dau vao
def process_data(X_data, y_data):
    X_data = np.array(X_data, dtype = 'float32') #chuyển X_data thành mảng numpy với kiểu là float32(số thực)
    if rgb:
        pass
    else:
        X_data = np.stack((X_data,)*3, axis=-1)
    # nếu rgb = true thì không làm gì cả, nhưng rgb có giá trị mặc định là false nên thực hiện trong else
    # trong else: xếp chồng X_data 3 lần theo trục cuối cùng, ở đây là 3 trục của ảnh RGB(chiều cao, chiều rộng, số kênh màu)
    # mục đích là để tạo ra 1 ảnh giả RGB từ hình ảnh đen trắng gốc
    X_data /= 255 
    # chia tất cả các giá trị trong X_data cho 255 để chuẩn hóa giá trị hình ảnh, vì giá trị trong pixel thô nằm trong khoảng từ 0 đến 255
    y_data = np.array(y_data) 
    # chuyển y_data thành mảng numpy 
    y_data = to_categorical(y_data)
    # chuyển y_data thành ma trận nhị phân, số lớp bằng số lớn nhất trong vector y_data + 1, data-type mặc định là float32
    return X_data, y_data

# Ham duuyet thu muc anh dung de train
def walk_file_tree(image_path):
    X_data = []
    y_data = []
    # 2 mảng trống để nhận dữ liệu X_data: hình ảnh, y_data: nhãn tương ứng với hình ảnh
    for directory, subdirectories, files in os.walk(image_path): 
    # duyệt qua tất cả thư mục trong image_path, ở đây là file data
        for file in files: 
        # duyệt từng file
            if not file.startswith('.'): # lấy các file không bắt đầu bằng '.' để xử lý
                path = os.path.join(directory, file) # tạo đường dẫn đầy đủ bằng cách nối tên thư mục + tên tệp hiện tại
                gesture_name = gestures[file[0:2]] # lấy ra 2 kí tự đầu của tên file rồi gọi phần tử tương ứng trong mảng gestures
                print(gesture_name) # in ra để kiểm tra
                print(gestures_map[gesture_name]) # in ra phần tử trong gestures_map tương ứng với gesture_name
                y_data.append(gestures_map[gesture_name]) # thêm giá trị gestures_map[gesture_name] vào mảng y_data
                X_data.append(process_image(path)) # xử lý hình ảnh rồi cho vào mảng X_data

            else:
                continue
    # xử lý lại hình ảnh và nhãn qua process_data            
    X_data, y_data = process_data(X_data, y_data)
    return X_data, y_data




# Load du lieu vao X va Y
X_data, y_data = walk_file_tree(image_path)

# Phan chia du lieu train va test theo ty le 80/20
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.2, random_state=12, stratify=y_data)
# 20% dữ liệu cho tập kiểm thử, còn lại 80% huấn luyện 
# random_state=12 là 1 hàm random nhưng gán sẵn giá trị là 12 để đảm bảo dữ liệu gốc, đồng thời có thể tái tạo
# stratify= y_data để đảm bảo số các lớp trong kiểm thử và huấn luyện là giống nhau

# Tối ưu hóa quá trình mô hình máy học
model_checkpoint = ModelCheckpoint(filepath=models_path, save_best_only=True)
# lưu mô hình với hiệu suất tốt nhất thông qua đường dẫn models_path, ở đây là file saved_model.hdf5

early_stopping = EarlyStopping(monitor='val_acc',
                               min_delta=0,
                               patience=10,
                               verbose=1,
                               mode='auto',
                               restore_best_weights=True)
# để dừng quá trình huấn luyện khi không cải thiện được hiệu suất sau 1 số lượng epoch cố định thông qua tham số patience,
# ở đây là 10.

# Khoi tao model
model1 = VGG16(weights='imagenet', include_top=False, input_shape=(imageSize, imageSize, 3))
# Tạo mô hình VGG16 với các trọng số huấn luyện trên tập dữ liệu imagenet(1 cơ sở dữ liệu hình ảnh)
# include_top=False là cấu hình để không bao gồm lớp cuối cùng, tức là không lấy dữ liệu đầu ra   
# input_shape=(imageSize, imageSize, 3) chỉ định kích thước ảnh imageSize x imageSize và số kênh màu là 3(rgb) 

optimizer1 = optimizers.Adam()
# Tạo thuật toán tối ưu hóa Adam(1 thuật toán phổ biến sử dụng trong các mô hình học sâu)

base_model = model1

# Them cac lop ben tren
x = base_model.output # lấy đầu ra của base_model(mô hình VGG16)
x = Flatten()(x) # ép dữ liệu x thành 1 vector 1D
x = Dense(128, activation='relu', name='fc1')(x)
x = Dense(128, activation='relu', name='fc2')(x)
x = Dense(128, activation='relu', name='fc2a')(x)
x = Dense(128, activation='relu', name='fc3')(x)
 # thêm 4 lớp đầu ra 128 neuron với hàm kích hoạt ReLU vào mô hình với tên khác nhau
x = Dropout(0.5)(x)
# thêm 1 lớp dropout với tỷ lệ giữ 50% neuron(giúp ngăn quá tải và tăng độ tin cậy cho mô hình)
x = Dense(64, activation='relu', name='fc4')(x)
# thêm một lớp đầu ra khác có 64 neuron với hàm kích hoạt ReLU vào mô hình

predictions = Dense(5, activation='softmax')(x)
# thêm một lớp đầu ra cuối cùng có 5 neuron với hàm kích hoạt softmax vào mô hình

model = Model(inputs=base_model.input, outputs=predictions)
# tạo một mô hình Keras với đầu vào là đầu vào của mô hình cơ sở và đầu ra là dự đoán của mô hình

# Đóng băng các lớp dưới, chỉ train các lớp bên trên mình thêm vào
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
# biên dịch mô hình theo thuật toán Adam, hàm mất là categorical_crossentropy, accuracy là chỉ số đánh giá sử dụng

model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test), verbose=1,
          callbacks=[early_stopping, model_checkpoint])
# huấn luyện mô hình 
# mô hình được huấn luyện trong 50 epoch (huấn luyện qua toàn bộ tập dữ liệu huấn luyện 50 lần)
# verbose=1 để hiển thị thông tin trong quá trình huấn luyện
# callbacks=[early_stopping, model_checkpoint] các callback được sử dụng trong quá trình huấn luyện

# Luu model da train ra file
model.save('models/mymodel.h5')


