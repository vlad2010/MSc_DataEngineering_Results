#include "serialportreader.h"
#include <QSerialPortInfo>
#include <QCoreApplication>
#include <QRegularExpression>
#include <QDebug>

const int MAX_SERIAL_DATA_SIZE = 4096; // Limit to prevent buffer overflow

SerialPortReader::SerialPortReader(const QString& portName, QObject *parent) :
    QObject(parent)
{
    // Validate port name (CWE-20)
    QRegularExpression portNamePattern("^COM[0-9]+$");
    if (!portNamePattern.match(portName).hasMatch()) {
        qWarning() << "Invalid serial port name:" << portName;
        throw std::invalid_argument("Invalid serial port name");
    }

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
        // Avoid logging sensitive info in production (CWE-532)
        qDebug() << "Started reading from serial port";
    } else {
        qWarning() << "Could not open serial port:" << m_serialPort.errorString();
        // Optionally, emit a signal or throw an exception for higher-level handling
    }
}

void SerialPortReader::handleReadyRead()
{
    QByteArray data = m_serialPort.readAll();

    // Validate data size (CWE-20)
    if (data.size() > MAX_SERIAL_DATA_SIZE) {
        qWarning() << "Received data exceeds maximum allowed size, discarding";
        return;
    }

    // Optionally, further validate/sanitize data content here

    emit newData(data);
}

void SerialPortReader::handleError(QSerialPort::SerialPortError error)
{
    if (error == QSerialPort::ReadError) {
        qWarning() << "I/O error occurred while reading from serial port:" << m_serialPort.errorString();
        // Optionally, close the port or signal error to higher layers
    }
}