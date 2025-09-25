#include "serialportreader.h"
#include <QSerialPortInfo>
#include <QCoreApplication>

SerialPortReader::SerialPortReader(QObject *parent) :
    QObject(parent)
{
    m_serialPort.setPortName("COM8");
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
    emit newData(data);
}

void SerialPortReader::handleError(QSerialPort::SerialPortError error)
{
    if (error == QSerialPort::ReadError) {
        qDebug() << "An I/O error occurred while reading the data from port" << m_serialPort.portName() << ", error:" << m_serialPort.errorString();
    }
}
