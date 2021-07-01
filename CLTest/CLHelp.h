#pragma once
#include <fstream>
#include <sstream>
#include <map>
#include <list>
#include <vector>
#include <iostream>
#include <iomanip>
#define CL_HPP_ENABLE_EXCEPTIONS true
#define CL_HPP_CL_1_2_DEFAULT_BUILD true
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#include "CL/cl2.hpp"
using namespace std;

namespace dev
{
namespace help
{
enum class ClPlatformTypeEnum
{
	Unknown,
	Amd,
	Clover,
	Nvidia
};
enum class DeviceTypeEnum
{
	Unknown,
	Cpu,
	Gpu,
	Accelerator
};
enum class DeviceSubscriptionTypeEnum
{
	None,
	OpenCL,
	Cuda,
	Cpu

};

struct DeviceDescriptor
{
	DeviceTypeEnum type = DeviceTypeEnum::Unknown;
	DeviceSubscriptionTypeEnum subscriptionType = DeviceSubscriptionTypeEnum::None;

	string uniqueId;     // For GPUs this is the PCI ID
	cl_ulong totalMemory;  // Total memory available by device
	string name;         // Device Name

	bool clDetected;  // For OpenCL detected devices
	string clName;
	unsigned int clPlatformId;
	string clPlatformName;
	ClPlatformTypeEnum clPlatformType = ClPlatformTypeEnum::Unknown;
	string clPlatformVersion;
	unsigned int clPlatformVersionMajor;
	unsigned int clPlatformVersionMinor;
	unsigned int clDeviceOrdinal;
	unsigned int clDeviceIndex;
	string clDeviceVersion;
	unsigned int clDeviceVersionMajor;
	unsigned int clDeviceVersionMinor;
	string clDriverVersion;
	string clBoardName;
	size_t clMaxMemAlloc;
	size_t clMaxWorkGroup;
	unsigned int clMaxComputeUnits;
	string clNvCompute;
	unsigned int clNvComputeMajor;
	unsigned int clNvComputeMinor;

	bool cuDetected;  // For CUDA detected devices
	string cuName;
	unsigned int cuDeviceOrdinal;
	unsigned int cuDeviceIndex;
	string cuCompute;
	unsigned int cuComputeMajor;
	unsigned int cuComputeMinor;

	int cpCpuNumer;   // For CPU
};

class CLHelp
{
public:
	CLHelp() {};
	~CLHelp() {};
	static void enumDevices(std::map<string, DeviceDescriptor>& _DevicesCollection);
};

} // namespase help
} // namespase dev