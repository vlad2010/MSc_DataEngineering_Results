#include "serialportreader.h"
#include <QSerialPortInfo>
#include <QCoreApplication>
#include <QSettings>

SerialPortReader::SerialPortReader(QObject *parent) :
    QObject(parent)
{
    // Load the port name from a configuration file instead of hardcoding
    QSettings settings("config.ini", QSettings::IniFormat);
    QString portName = settings.value("SerialPort/PortName", "COM8").toString();
    
    m_serialPort.setPortName(portName);
    m_serialPort.setBaudRate(QSerialPort::Baud115200);

    connect(&m_serialPort, &QSerialPort::readyRead, this, &SerialPortReader::handleReadyRead);
    connect(&m_serialPort, &QSerialPort::errorOccurred, this, &SerialPortReader::handleError);
}

SerialPortReader::~SerialPortReader()
{
    if (m_serialPort.isOpen())
        m_serialPort.close();
}

void SerialPortReader::start()
{
    if (m_serialPort.open(QIODevice::ReadOnly)) {
        qDebug() << "Started reading from" << m_serialPort.portName();
    } else {
        qDebug() << "Could not open port" << m_serialPort.portName() << ":" << m_serialPort.errorString();
    }
}

void SerialPortReader::handleReadyRead()
{
    QByteArray data = m_serialPort.readAll();
    
    // Validate the data (this would be specific to the application needs)
    if (isDataValid(data)) {
        emit newData(data);
    } else {
        qDebug() << "Received invalid data";
    }
}

// Add a function to check data validity
bool SerialPortReader::isDataValid(const QByteArray& data)
{
    // Implement validation logic as per the expected data format
    return !data.isEmpty(); // Example validation
}

void SerialPortReader::handleError(QSerialPort::SerialPortError error)
{
    switch (error) {
    case QSerialPort::ReadError:
        qDebug() << "An I/O error occurred while reading the data from port" << m_serialPort.portName() << ", error:" << m_serialPort.errorString();
        break;
    case QSerialPort::ResourceError:
        qDebug() << "Resource error on port" << m_serialPort.portName() << ", closing port.";
        m_serialPort.close();
        break;
    case QSerialPort::PermissionError:
        qDebug() << "Permission error on port" << m_serialPort.portName() << ", check permissions.";
        break;
    default:
        qDebug() << "An error occurred on port" << m_serialPort.portName() << ", error:" << m_serialPort.errorString();
        break;
    }
}