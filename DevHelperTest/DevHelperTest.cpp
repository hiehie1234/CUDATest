// DevHelperTest.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"

#include  "..\DevHelper\DevHelper.h"
#define ETH_ETHASHCL true
#define ETH_ETHASHCUDA true;

using namespace dev::help;
const UINT WM_UPDATE_DEVICEDETECTED = ::RegisterWindowMessage(L"DeviceDetected");  // jie 20210205
int main()
{
	// Physical Mining Devices descriptor
	std::map<std::string, DeviceDescriptor> m_DevicesCollection = {};
	// Mining options
	MinerType m_minerType = MinerType::Mixed;
	bool m_Detected = false;
#if ETH_ETHASHCL
	if (m_minerType == MinerType::CL || m_minerType == MinerType::Mixed)
		CLHelp::enumDevices(m_DevicesCollection);
#endif
#if ETH_ETHASHCUDA
	if (m_minerType == MinerType::CUDA || m_minerType == MinerType::Mixed)
		CUDAHelp::enumDevices(m_DevicesCollection);
#endif

	if (m_DevicesCollection.size() == 0) {
		cout << ("There are no available device(s) that support") << endl;
	}
	else {
		printf("Detected %d Capable device(s)\n", m_DevicesCollection.size());
	}
	for (auto it = m_DevicesCollection.begin(); it != m_DevicesCollection.end(); it++)
	{
		
		if (it->second.clDetected || it->second.cuDetected) 
		{
			if (it->second.totalMemory / 1024 / 1024 / 1024 > 4)
			{
				m_Detected = true;
			}
		}
		cout << "namess: " << it->second.name << endl;
		cout << "totalMemory:" << it->second.totalMemory << endl;
	}
	if(m_Detected)
		::PostMessage(HWND_BROADCAST, WM_UPDATE_DEVICEDETECTED, 0, 0);
	else
		::PostMessage(HWND_BROADCAST, WM_UPDATE_DEVICEDETECTED, -1, 0);
    return 0;
}

