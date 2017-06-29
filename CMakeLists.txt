cmake_minimum_required(VERSION 2.8.4)
project(FER)

add_compile_options(-std=c++11)
add_compile_options(-O3)

set(FER_needed_libraries)
set(FER_INCLUDE_DIRS)

find_package(OpenCV REQUIRED)
if (OpenCV_FOUND)
	set (FER_needed_libraries ${FER_needed_libraries} ${OpenCV_LIBS})
	set (FER_INCLUDE_DIRS ${FER_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
else()
   message("OpenCV not found, so we won't build the webcam_face_pose_ex example.")
endif()

find_package(dlib REQUIRED)
if (dlib_FOUND)
	set (FER_needed_libraries ${FER_needed_libraries} ${dlib_LIBS})
	set (FER_INCLUDE_DIRS ${FER_INCLUDE_DIRS} ${dlib_INCLUDE_DIRS})
else()
   message("dlib not found, so we won't build the webcam_face_pose_ex example.")
endif()

include_directories(${FER_INCLUDE_DIRS})

# ADD_EXECUTABLE(show main.cpp fer.cpp)
# TARGET_LINK_LIBRARIES(show ${FER_needed_libraries})

ADD_EXECUTABLE(gather gather.cpp fer.cpp)
TARGET_LINK_LIBRARIES(gather ${FER_needed_libraries})

ADD_EXECUTABLE(fer_train fer_train.cpp fer.cpp)
TARGET_LINK_LIBRARIES(fer_train ${FER_needed_libraries})

ADD_EXECUTABLE(fer_predict fer_predict.cpp fer.cpp)
TARGET_LINK_LIBRARIES(fer_predict ${FER_needed_libraries})

ADD_EXECUTABLE(fer_train_CK train_CK.cpp fer.cpp)
TARGET_LINK_LIBRARIES(fer_train_CK ${FER_needed_libraries})

ADD_EXECUTABLE(fer_train_jaffe train_jaffe.cpp fer.cpp)
TARGET_LINK_LIBRARIES(fer_train_jaffe ${FER_needed_libraries})

