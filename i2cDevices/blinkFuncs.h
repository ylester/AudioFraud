#include <iostream>
#include <errno.h>
#include <wiringPiI2C.h>

#define setNow 110

void setRGB ( int fd,int red , int green, int blue)
{
		//command = 
		wiringPiI2CWrite(fd, setNow);
		//red = 
		wiringPiI2CWrite(fd,red);
		//green = 
		wiringPiI2CWrite(fd,blue);
		//blue = 
		wiringPiI2CWrite(fd,green);
}

void LEDOFF ( int fd)
{
		//command = 
		wiringPiI2CWrite(fd, setNow);
		//red = 
		wiringPiI2CWrite(fd,0);
		//green = 
		wiringPiI2CWrite(fd,0);
		//blue = 
		wiringPiI2CWrite(fd,0);
}

