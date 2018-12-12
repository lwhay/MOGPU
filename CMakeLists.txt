cmake_minimum_required(VERSION 3.12)
project(MOGPU LANGUAGES C CXX CUDA)

#set(CMAKE_CXX_STANDARD 14)

find_package(Threads)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --cudart static --relocatable-device-code=true -gencode arch=compute_35,code=compute_35 -gencode arch=compute_35,code=sm_35")

include_directories(.)
include_directories(kernel)
include_directories(misc)
include_directories(config)
include_directories(basis)
include_directories(tinyxml)

add_executable(DEVDETECT
        basis/kernel.cu
        basis/main.cu)

add_executable(MOGPU
        basis/kernel.h
        #basis/main.cu
        tinyxml/tinystr.cpp
        tinyxml/tinystr.h
        tinyxml/tinyxml.cpp
        tinyxml/tinyxml.h
        tinyxml/tinyxmlerror.cpp
        tinyxml/tinyxmlparser.cpp
        misc/BaseStruct.h
        misc/Buffer.h
        misc/Cell.cu
        misc/Cell.cuh
        config/Config.cu
        config/Config.h
        device/DeriveStruct.h
        device/DeviceGlobalVar.cu
        device/DeviceGlobalVar.cuh
        kernel/DistributorKernel.cu
        kernel/DistributorKernel.cuh
        config/GConfig.cu
        config/GConfig.h
        misc/GPGrid.cu
        misc/GPGrid.h
        misc/Grid.cu
        misc/Grid.cuh
        kernel/InitKernel.cu
        kernel/InitKernel.cuh
        misc/MemnodePool.cuh
        misc/ObjBox.cu
        misc/ObjBox.cuh
        misc/QueryCacheArea.h
        kernel/QueryKernel.cu
        kernel/QueryKernel.cuh
        misc/QueryQNode.h
        misc/Rect.cuh
        misc/SecIndex.cuh
        misc/SIEntry.cu
        misc/SIEntry.cuh
        misc/SyncFuncGPU.cu
        misc/SyncFuncGPU.cuh
        misc/UpdateCacheArea.cuh
        kernel/UpdateKernel.cu
        kernel/UpdateKernel.cuh
        misc/UpdateQNode.cuh
        main.cpp)
