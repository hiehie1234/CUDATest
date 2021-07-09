// DevHelper.cpp : 定义 DLL 应用程序的导出函数。
//

#include "stdafx.h"

#define CL_HPP_ENABLE_EXCEPTIONS true
#define CL_HPP_CL_1_2_DEFAULT_BUILD true
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#include "CL/cl2.hpp"
#include "DevHelper.h"
#include "cuda_runtime.h"

namespace dev
{
namespace help
{
int getVersion()
{
	return 11120;
};

std::vector<cl::Platform> getPlatforms()
{
	vector<cl::Platform> platforms;
	try
	{
		cl::Platform::get(&platforms);
	}
	catch (cl::Error const& err)
	{
#if defined(CL_PLATFORM_NOT_FOUND_KHR)
		if (err.err() == CL_PLATFORM_NOT_FOUND_KHR)
			std::cerr << "No OpenCL platforms found" << std::endl;
		else
#endif
			std::cerr << "OpenCL error : " << err.what();
	}
	return platforms;
}
std::vector<cl::Device> getDevices(
	std::vector<cl::Platform> const& _platforms, unsigned _platformId)
{
	vector<cl::Device> devices;
	size_t platform_num = min<size_t>(_platformId, _platforms.size() - 1);
	try
	{
		_platforms[platform_num].getDevices(
			CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_ACCELERATOR, &devices);
	}
	catch (cl::Error const& err)
	{
		// if simply no devices found return empty vector
		if (err.err() != CL_DEVICE_NOT_FOUND)
			throw err;
	}
	return devices;
}

void CLHelp::enumDevices(std::map<string, DeviceDescriptor>& _DevicesCollection)
{
	// Load available platforms
	vector<cl::Platform> platforms = getPlatforms();
	if (platforms.empty())
		return;

	unsigned int dIdx = 0;
	for (unsigned int pIdx = 0; pIdx < platforms.size(); pIdx++)
	{
		std::string platformName = platforms.at(pIdx).getInfo<CL_PLATFORM_NAME>();
		ClPlatformTypeEnum platformType = ClPlatformTypeEnum::Unknown;
		if (platformName == "AMD Accelerated Parallel Processing")
			platformType = ClPlatformTypeEnum::Amd;
		else if (platformName == "Clover")
			platformType = ClPlatformTypeEnum::Clover;
		else if (platformName == "NVIDIA CUDA")
			platformType = ClPlatformTypeEnum::Nvidia;
		else
		{
			std::cerr << "Unrecognized platform " << platformName << std::endl;
			continue;
		}


		std::string platformVersion = platforms.at(pIdx).getInfo<CL_PLATFORM_VERSION>();
		unsigned int platformVersionMajor = std::stoi(platformVersion.substr(7, 1));
		unsigned int platformVersionMinor = std::stoi(platformVersion.substr(9, 1));

		dIdx = 0;
		vector<cl::Device> devices = getDevices(platforms, pIdx);
		for (auto const& device : devices)
		{
			DeviceTypeEnum clDeviceType = DeviceTypeEnum::Unknown;
			cl_device_type detectedType = device.getInfo<CL_DEVICE_TYPE>();
			if (detectedType == CL_DEVICE_TYPE_GPU)
				clDeviceType = DeviceTypeEnum::Gpu;
			else if (detectedType == CL_DEVICE_TYPE_CPU)
				clDeviceType = DeviceTypeEnum::Cpu;
			else if (detectedType == CL_DEVICE_TYPE_ACCELERATOR)
				clDeviceType = DeviceTypeEnum::Accelerator;

			string uniqueId;
			DeviceDescriptor deviceDescriptor;

			if (clDeviceType == DeviceTypeEnum::Gpu && platformType == ClPlatformTypeEnum::Nvidia)
			{
				cl_int bus_id, slot_id;
				if (clGetDeviceInfo(device.get(), 0x4008, sizeof(bus_id), &bus_id, NULL) ==
					CL_SUCCESS &&
					clGetDeviceInfo(device.get(), 0x4009, sizeof(slot_id), &slot_id, NULL) ==
					CL_SUCCESS)
				{
					std::ostringstream s;
					s << setfill('0') << setw(2) << hex << bus_id << ":" << setw(2)
						<< (unsigned int)(slot_id >> 3) << "." << (unsigned int)(slot_id & 0x7);
					uniqueId = s.str();
				}
			}
			else if (clDeviceType == DeviceTypeEnum::Gpu &&
				(platformType == ClPlatformTypeEnum::Amd ||
					platformType == ClPlatformTypeEnum::Clover))
			{
				cl_char t[24];
				if (clGetDeviceInfo(device.get(), 0x4037, sizeof(t), &t, NULL) == CL_SUCCESS)
				{
					std::ostringstream s;
					s << setfill('0') << setw(2) << hex << (unsigned int)(t[21]) << ":" << setw(2)
						<< (unsigned int)(t[22]) << "." << (unsigned int)(t[23]);
					uniqueId = s.str();
				}
			}
			else if (clDeviceType == DeviceTypeEnum::Cpu)
			{
				std::ostringstream s;
				s << "CPU:" << setfill('0') << setw(2) << hex << (pIdx + dIdx);
				uniqueId = s.str();
			}
			else
			{
				// We're not prepared (yet) to handle other platforms or types
				++dIdx;
				continue;
			}

			if (_DevicesCollection.find(uniqueId) != _DevicesCollection.end())
				deviceDescriptor = _DevicesCollection[uniqueId];
			else
				deviceDescriptor = DeviceDescriptor();

			// Fill the blanks by OpenCL means
			deviceDescriptor.name = device.getInfo<CL_DEVICE_NAME>();
			deviceDescriptor.type = clDeviceType;
			deviceDescriptor.uniqueId = uniqueId;
			deviceDescriptor.clDetected = true;
			deviceDescriptor.clPlatformId = pIdx;
			deviceDescriptor.clPlatformName = platformName;
			deviceDescriptor.clPlatformType = platformType;
			deviceDescriptor.clPlatformVersion = platformVersion;
			deviceDescriptor.clPlatformVersionMajor = platformVersionMajor;
			deviceDescriptor.clPlatformVersionMinor = platformVersionMinor;
			deviceDescriptor.clDeviceOrdinal = dIdx;

			deviceDescriptor.clName = deviceDescriptor.name;
			deviceDescriptor.clDeviceVersion = device.getInfo<CL_DEVICE_VERSION>();
			deviceDescriptor.clDeviceVersionMajor =
				std::stoi(deviceDescriptor.clDeviceVersion.substr(7, 1));
			deviceDescriptor.clDeviceVersionMinor =
				std::stoi(deviceDescriptor.clDeviceVersion.substr(9, 1));
			deviceDescriptor.totalMemory = device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
			deviceDescriptor.clMaxMemAlloc = device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();
			deviceDescriptor.clMaxWorkGroup = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
			deviceDescriptor.clMaxComputeUnits = device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
			// Apparently some 36 CU devices return a bogus 14!!!
			deviceDescriptor.clMaxComputeUnits =
				deviceDescriptor.clMaxComputeUnits == 14 ? 36 : deviceDescriptor.clMaxComputeUnits;

			// Is it an NVIDIA card ?
			if (platformType == ClPlatformTypeEnum::Nvidia)
			{
				size_t siz;
				clGetDeviceInfo(device.get(), CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV,
					sizeof(deviceDescriptor.clNvComputeMajor), &deviceDescriptor.clNvComputeMajor,
					&siz);
				clGetDeviceInfo(device.get(), CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV,
					sizeof(deviceDescriptor.clNvComputeMinor), &deviceDescriptor.clNvComputeMinor,
					&siz);
				deviceDescriptor.clNvCompute = to_string(deviceDescriptor.clNvComputeMajor) + "." +
					to_string(deviceDescriptor.clNvComputeMinor);
			}

			// Upsert Devices Collection
			_DevicesCollection[uniqueId] = deviceDescriptor;
			++dIdx;

		}
	}
};

int CUDAHelp::getNumDevices()
{
	int deviceCount;
	cudaError_t err = cudaGetDeviceCount(&deviceCount);
	if (err == cudaSuccess)
		return deviceCount;

	if (err == cudaErrorInsufficientDriver)
	{
		int driverVersion = 0;
		cudaDriverGetVersion(&driverVersion);
		if (driverVersion == 0)
			std::cerr << "CUDA Error : No CUDA driver found" << std::endl;
		else
			std::cerr << "CUDA Error : Insufficient CUDA driver " << std::to_string(driverVersion)
			<< std::endl;
	}
	else
	{
		std::cerr << "CUDA Error : " << cudaGetErrorString(err) << std::endl;
	}

	return 0;
}
void CUDAHelp::enumDevices(std::map<string, DeviceDescriptor>& _DevicesCollection)
{
	int numDevices = getNumDevices();

	for (int i = 0; i < numDevices; i++)
	{
		string uniqueId;
		ostringstream s;
		DeviceDescriptor deviceDescriptor;
		cudaDeviceProp props;

		try
		{
			size_t freeMem, totalMem;
			CUDA_SAFE_CALL(cudaGetDeviceProperties(&props, i));
			CUDA_SAFE_CALL(cudaMemGetInfo(&freeMem, &totalMem));
			s << setw(2) << setfill('0') << hex << props.pciBusID << ":" << setw(2)
				<< props.pciDeviceID << ".0";
			uniqueId = s.str();

			if (_DevicesCollection.find(uniqueId) != _DevicesCollection.end())
				deviceDescriptor = _DevicesCollection[uniqueId];
			else
				deviceDescriptor = DeviceDescriptor();

			deviceDescriptor.name = string(props.name);
			deviceDescriptor.cuDetected = true;
			deviceDescriptor.uniqueId = uniqueId;
			deviceDescriptor.type = DeviceTypeEnum::Gpu;
			deviceDescriptor.cuDeviceIndex = i;
			deviceDescriptor.cuDeviceOrdinal = i;
			deviceDescriptor.cuName = string(props.name);
			deviceDescriptor.totalMemory = props.totalGlobalMem;
			deviceDescriptor.cuCompute =
				(to_string(props.major) + "." + to_string(props.minor));
			deviceDescriptor.cuComputeMajor = props.major;
			deviceDescriptor.cuComputeMinor = props.minor;

			_DevicesCollection[uniqueId] = deviceDescriptor;
		}
		catch (const cuda_runtime_error& _e)
		{
			std::cerr << _e.what() << std::endl;
		}
	}
}
} // namespace dev
} // namespace help