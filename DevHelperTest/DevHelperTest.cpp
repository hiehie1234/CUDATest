// DevHelperTest.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"

#include "C:\Users\jiewu\Documents\Visual Studio 2015\Projects\CUDATest\DevHelper\DevHelper.h"

using namespace dev::clhelp;
int main()
{
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

		cout << "namess: " << it->second.name << endl;
		cout << "totalMemory:" << it->second.totalMemory << endl;
	}
    return 0;
}

