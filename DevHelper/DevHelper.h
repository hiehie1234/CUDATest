#pragma once
#include <sstream>
#include <map>
#include <iostream>
#include <iomanip>

using namespace std;
#ifdef __cplusplus
extern "C" {
#endif

#define DEVHELPERDLL_EXPORTS

#ifdef DEVHELPERDLL_EXPORTS
#define DEVHELPERDLL_API _declspec(dllexport)
#else
#define DEVHELPERDLL_API __declspec(dllimport)
#endif
namespace dev
{
namespace clhelp
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
	_ULonglong totalMemory;  // Total memory available by device
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

DEVHELPERDLL_API int getVersion();
class DEVHELPERDLL_API CLHelp {
public:
	CLHelp(void);
	~CLHelp(void);
	static void enumDevices(std::map<string, DeviceDescriptor>& _DevicesCollection);
};
} // namespace help
} // namespace dev

#ifdef __cplusplus
}
#endif