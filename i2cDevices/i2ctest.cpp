#include <iostream>
#include <errno.h>
#include <wiringPiI2C.h>
#include <cstring>
#include "blinkFuncs.h"
#include "LCDFuncs.h"


using namespace std;

//SerLCD lcd; // initialize the LCD library with default I2C address of 0x72

#define blinkAddress 0x09
#define LCDAddress 0x72

int main ()
{
	int fd1, fd2, red,green,blue, strLength;
	fd1 = wiringPiI2CSetup(blinkAddress);
	fd2 = wiringPiI2CSetup(LCDAddress);
	cout << "init result: "<< fd1 << " " << fd2 <<endl;
	LCDSetBacklight(fd2, 255, 255, 255);
	LCDSetContrast(fd2 , 0);
	LCDClear(fd2);
	LEDOFF(fd1);
	LCDOFF(fd2);
	//setRGB ( fd1 , 255, 0, 0);
	//strLength = LCDPrintString( fd2, "Hello World");
}
