#include <iostream>
#include <errno.h>
#include <wiringPiI2C.h>
#include <cstring>


#define DISPLAY_ADDRESS1 0x72 //This is the default address of the OpenLCD
#define MAX_ROWS 4
#define MAX_COLUMNS 20

//OpenLCD command characters
#define SPECIAL_COMMAND 254  //Magic number for sending a special command
#define SETTING_COMMAND 124  // ,|, the pipe character: The command to change settings: baud, lines, width, backlight, splash, etc

//OpenLCD commands
#define CLEAR_COMMAND 45					//45, -, the dash character: command to clear and home the display
#define CONTRAST_COMMAND 0x18				//Command to change the contrast setting
#define ADDRESS_COMMAND 0x19				//Command to change the i2c address
#define SET_RGB_COMMAND 43					// +, the plus character: command to set backlight RGB value
#define ENABLE_SYSTEM_MESSAGE_DISPLAY 0x2E  //46, ., command to enable system messages being displayed
#define DISABLE_SYSTEM_MESSAGE_DISPLAY 0x2F //47, /, command to disable system messages being displayed
#define ENABLE_SPLASH_DISPLAY 0x30			//48, 0, command to enable splash screen at power on
#define DISABLE_SPLASH_DISPLAY 0x31			//49, 1, command to disable splash screen at power on
#define SAVE_CURRENT_DISPLAY_AS_SPLASH 0x0A //10, Ctrl+j, command to save current text on display as splash

// special commands
#define LCD_RETURNHOME 0x02
#define LCD_ENTRYMODESET 0x04
#define LCD_DISPLAYCONTROL 0x08
#define LCD_CURSORSHIFT 0x10
#define LCD_SETDDRAMADDR 0x80

// flags for display entry mode
#define LCD_ENTRYRIGHT 0x00
#define LCD_ENTRYLEFT 0x02
#define LCD_ENTRYSHIFTINCREMENT 0x01
#define LCD_ENTRYSHIFTDECREMENT 0x00

// flags for display on/off control
#define LCD_DISPLAYON 0x04
#define LCD_DISPLAYOFF 0
#define LCD_CURSORON 0x02
#define LCD_CURSOROFF 0x00
#define LCD_BLINKON 0x01
#define LCD_BLINKOFF 0x00

void LCDOFF(int fd)
{
	wiringPiI2CWrite(fd, LCD_DISPLAYOFF);
}


int LCDPrintString ( int fd , const char *str)
{
	if (str == NULL)
		return 0;
	else
	{
		size_t strLength = strlen(str);
		int n=0;
		while(strLength--)
		{
			wiringPiI2CWrite(fd, uint8_t(*str++));
			n++;
		}
		return n;
	}
}

void LCDSetBacklight ( int fd, int red , int green, int blue)
{
	wiringPiI2CWrite(fd, SETTING_COMMAND); // writing a setting commandd
	wiringPiI2CWrite (fd ,SET_RGB_COMMAND); //to set an RGB backlight value
	wiringPiI2CWrite (fd ,red);
	wiringPiI2CWrite (fd ,green); 
	wiringPiI2CWrite (fd ,blue); 
}

void LCDSetContrast ( int fd, int newValue)
{
	wiringPiI2CWrite(fd, SETTING_COMMAND); // writing a setting commandd
	wiringPiI2CWrite (fd ,CONTRAST_COMMAND); //to set COntrast value
	wiringPiI2CWrite (fd ,newValue);
}

void LCDClear ( int fd)
{
	wiringPiI2CWrite(fd, SETTING_COMMAND); // writing a setting commandd
	wiringPiI2CWrite(fd, CLEAR_COMMAND); // writing a setting commandd
}

