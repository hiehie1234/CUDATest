// CLTest.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include "CLHelp.h"


using namespace std;
using namespace dev::help;

//#include "C:\Users\jiewu\Documents\Visual Studio 2015\Projects\CUDATest\DevHelper\DevHelper.h"
//using namespace dev::clhelp;

int main()
{
	///* Host/device data structures */
	//cl_platform_id *platforms;
	//cl_device_id *devices;
	//cl_uint num_platforms;
	//cl_uint num_devices, addr_data;
	//cl_int i, err;

	//err = clGetPlatformIDs(5, NULL, &num_platforms);
	//if (err < 0) {
	//	perror("Couldn't find any platforms.");
	//	exit(1);
	//}
	///* 选取全部的platforms*/
	//platforms = (cl_platform_id*)
	//	malloc(sizeof(cl_platform_id) * num_platforms);
	//err = clGetPlatformIDs(num_platforms, platforms, NULL);
	//if (err < 0) {
	//	perror("Couldn't find any platforms");
	//	exit(1);
	//}

	

	//clGetDeviceInfo(ctx.DeviceID, CL_DEVICE_VENDOR_ID, sizeof(size_t), &(info), nullptr);
	//printf("CL_DEVICE_VENDOR_ID:%d\n", info);
	//clGetDeviceInfo(ctx.DeviceID, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(size_t), &(info), nullptr);
	//printf("CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS:%d\n", info);
	//clGetDeviceInfo(ctx.DeviceID, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &(info), nullptr);
	//printf("CL_DEVICE_MAX_WORK_GROUP_SIZE:%d\n", info);
	//clGetDeviceInfo(ctx.DeviceID, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t), &(info), nullptr);
	//printf("CL_DEVICE_MAX_WORK_ITEM_SIZES:%d\n", info);
	//clGetDeviceInfo(ctx.DeviceID, CL_DEVICE_LOCAL_MEM_TYPE, sizeof(size_t), &(info), nullptr);
	//printf("CL_DEVICE_LOCAL_MEM_SIZE:%d\n", info);
	//clGetDeviceInfo(ctx.DeviceID, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(size_t), &(info), nullptr);
	//printf("CL_DEVICE_LOCAL_MEM_SIZE:%d\n", info);
	//clGetDeviceInfo(ctx.DeviceID, CL_DEVICE_GLOBAL_MEM_CACHE_TYPE, sizeof(size_t), &(info), nullptr);
	//printf("CL_DEVICE_GLOBAL_MEM_CACHE_TYPE:%d\n", info);
	//clGetDeviceInfo(ctx.DeviceID, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, sizeof(size_t), &(info), nullptr);
	//printf("CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE:%d\n", info);
	//clGetDeviceInfo(ctx.DeviceID, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(size_t), &(info), nullptr);
	//printf("CL_DEVICE_GLOBAL_MEM_CACHE_SIZE:%d\n", info);
	//clGetDeviceInfo(ctx.DeviceID, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(size_t), &(info), nullptr);
	//printf("CL_DEVICE_GLOBAL_MEM_SIZE:%d\n", info);
	//clGetDeviceInfo(ctx.DeviceID, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(size_t), &(info), nullptr);
	//printf("CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE:%lld\n", info);
	//clGetDeviceInfo(ctx.DeviceID, CL_DEVICE_MAX_CONSTANT_ARGS, sizeof(size_t), &(info), nullptr);
	//printf("CL_DEVICE_MAX_CONSTANT_ARGS:%d\n", info);

	
	// Physical Mining Devices descriptor
	std::map<std::string, DeviceDescriptor> m_DevicesCollection = {};

	CLHelp::enumDevices(m_DevicesCollection);
	
	if (m_DevicesCollection.size() == 0) {
		cout << ("There are no available device(s) that support") << endl;
	}
	else {
		printf("Detected %d Capable device(s)\n", m_DevicesCollection.size());
	}
	for (auto it = m_DevicesCollection.begin(); it != m_DevicesCollection.end(); it++)
	{
		/*if (!it->second.cuDetected ||
			it->second.subscriptionType != DeviceSubscriptionTypeEnum::None)
			continue;
		*///it->second.subscriptionType = DeviceSubscriptionTypeEnum::Cuda;
		
		cout << "platformName:" << it->second.clPlatformName << endl;
		//cout << "platFormType:" << it->second.clPlatformTyp << endl;
		cout << "name: " << it->second.clDriverVersion << endl;
		cout << "totalMemory:" << it->second.clDeviceVersion << endl;
	}
	
    return 0;
}

