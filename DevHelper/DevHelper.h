#pragma once

#include "Help.h"
#include <sstream>

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
namespace help
{


DEVHELPERDLL_API int getVersion();

class DEVHELPERDLL_API CLHelp : public Help {
public:
	CLHelp(void) {};
	~CLHelp(void) {};
	static void enumDevices(std::map<string, DeviceDescriptor>& _DevicesCollection);
};
class DEVHELPERDLL_API CUDAHelp : public Help {
public:
	CUDAHelp(void) {};
	~CUDAHelp(void) {};
	static int getNumDevices();
	static void enumDevices(std::map<string, DeviceDescriptor>& _DevicesCollection);
};
} // namespace help
} // namespace dev

#ifdef __cplusplus
}
#endif