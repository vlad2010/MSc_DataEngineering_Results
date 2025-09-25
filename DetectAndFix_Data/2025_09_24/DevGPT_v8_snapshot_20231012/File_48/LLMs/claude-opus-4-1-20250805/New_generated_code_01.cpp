#include "serialportreader.h"
#include <QSerialPortInfo>
#include <QCoreApplication>
#include <QTimer>
#include <QMutex>
#include <QMutexLocker>

class SerialPortReader : public QObject
{
    Q_OBJECT

public:
    explicit SerialPortReader(const QString& portName = QString(), QObject *parent = nullptr);
    ~SerialPortReader();
    
    bool setPortName(const QString& portName);
    void start();
    void stop();
    
signals:
    void newData(const QByteArray& data);
    void errorOccurred(const QString& error);
    
private slots:
    void handleReadyRead();
    void handleError(QSerialPort::SerialPortError error);
    void handleTimeout();
    
private:
    static constexpr int MAX_READ_SIZE = 4096;  // Maximum bytes to read at once
    static constexpr int READ_TIMEOUT_MS = 5000; // Read timeout in milliseconds
    static constexpr int MAX_BUFFER_SIZE = 65536; // Maximum buffer size (64KB)
    
    QSerialPort m_serialPort;
    QTimer* m_timeoutTimer;
    QByteArray m_buffer;
    QMutex m_mutex;
    bool m_isRunning;
    
    bool validatePortName(const QString& portName);
    void clearBuffer();
};

SerialPortReader::SerialPortReader(const QString& portName, QObject *parent) :
    QObject(parent),
    m_timeoutTimer(new QTimer(this)),
    m_isRunning(false)
{
    // Configure serial port with secure defaults
    m_serialPort.setBaudRate(QSerialPort::Baud115200);
    m_serialPort.setDataBits(QSerialPort::Data8);
    m_serialPort.setParity(QSerialPort::NoParity);
    m_serialPort.setStopBits(QSerialPort::OneStop);
    m_serialPort.setFlowControl(QSerialPort::NoFlowControl);
    
    // Set timeouts for read/write operations
    m_serialPort.setReadBufferSize(MAX_BUFFER_SIZE);
    
    // Validate and set port name if provided
    if (!portName.isEmpty()) {
        if (!setPortName(portName)) {
            qWarning() << "Invalid port name provided:" << portName;
        }
    }
    
    // Setup connections with proper error handling
    connect(&m_serialPort, &QSerialPort::readyRead, 
            this, &SerialPortReader::handleReadyRead);
    connect(&m_serialPort, &QSerialPort::errorOccurred, 
            this, &SerialPortReader::handleError);
    
    // Setup timeout timer for detecting stalled reads
    m_timeoutTimer->setInterval(READ_TIMEOUT_MS);
    m_timeoutTimer->setSingleShot(true);
    connect(m_timeoutTimer, &QTimer::timeout, 
            this, &SerialPortReader::handleTimeout);
}

SerialPortReader::~SerialPortReader()
{
    stop();
}

bool SerialPortReader::validatePortName(const QString& portName)
{
    if (portName.isEmpty()) {
        return false;
    }
    
    // Check if port exists in available ports list
    const auto availablePorts = QSerialPortInfo::availablePorts();
    for (const QSerialPortInfo &info : availablePorts) {
        if (info.portName() == portName || 
            info.systemLocation() == portName) {
            return true;
        }
    }
    
    return false;
}

bool SerialPortReader::setPortName(const QString& portName)
{
    QMutexLocker locker(&m_mutex);
    
    // Don't change port while running
    if (m_isRunning) {
        qWarning() << "Cannot change port while reader is running";
        return false;
    }
    
    // Validate port name
    if (!validatePortName(portName)) {
        qWarning() << "Port" << portName << "is not available";
        emit errorOccurred(QString("Port %1 is not available").arg(portName));
        return false;
    }
    
    m_serialPort.setPortName(portName);
    return true;
}

void SerialPortReader::start()
{
    QMutexLocker locker(&m_mutex);
    
    if (m_isRunning) {
        qDebug() << "Serial port reader is already running";
        return;
    }
    
    // Validate port name before opening
    if (m_serialPort.portName().isEmpty()) {
        qCritical() << "No port name specified";
        emit errorOccurred("No port name specified");
        return;
    }
    
    if (!validatePortName(m_serialPort.portName())) {
        qCritical() << "Port" << m_serialPort.portName() << "is not available";
        emit errorOccurred(QString("Port %1 is not available").arg(m_serialPort.portName()));
        return;
    }
    
    // Clear any existing buffer data
    clearBuffer();
    
    // Attempt to open the port with error handling
    if (m_serialPort.open(QIODevice::ReadOnly)) {
        m_isRunning = true;
        qDebug() << "Started reading from" << m_serialPort.portName()
                 << "with buffer size limit:" << MAX_BUFFER_SIZE;
    } else {
        qCritical() << "Could not open port" << m_serialPort.portName() 
                    << ":" << m_serialPort.errorString();
        emit errorOccurred(QString("Could not open port %1: %2")
                          .arg(m_serialPort.portName())
                          .arg(m_serialPort.errorString()));
    }
}

void SerialPortReader::stop()
{
    QMutexLocker locker(&m_mutex);
    
    if (m_timeoutTimer->isActive()) {
        m_timeoutTimer->stop();
    }
    
    if (m_serialPort.isOpen()) {
        m_serialPort.close();
    }
    
    clearBuffer();
    m_isRunning = false;
    
    qDebug() << "Stopped reading from serial port";
}

void SerialPortReader::handleReadyRead()
{
    QMutexLocker locker(&m_mutex);
    
    if (!m_isRunning) {
        return;
    }
    
    // Reset timeout timer
    m_timeoutTimer->stop();
    m_timeoutTimer->start();
    
    // Read data with size limit to prevent memory exhaustion
    qint64 bytesAvailable = m_serialPort.bytesAvailable();
    
    if (bytesAvailable <= 0) {
        return;
    }
    
    // Limit the amount of data read at once
    qint64 bytesToRead = qMin(bytesAvailable, static_cast<qint64>(MAX_READ_SIZE));
    
    // Check if buffer would exceed maximum size
    if (m_buffer.size() + bytesToRead > MAX_BUFFER_SIZE) {
        qWarning() << "Buffer size limit reached, clearing buffer";
        clearBuffer();
        emit errorOccurred("Buffer overflow - data loss occurred");
    }
    
    // Read the data
    QByteArray data = m_serialPort.read(bytesToRead);
    
    if (data.isEmpty()) {
        return;
    }
    
    // Validate data (basic validation - can be extended based on protocol)
    // For example, checking for non-printable characters if expecting text
    bool hasValidData = true;
    for (unsigned char byte : data) {
        // Example: reject data with certain control characters
        if (byte == 0x00) {  // NULL character
            hasValidData = false;
            qWarning() << "Received invalid data (NULL bytes detected)";
            break;
        }
    }
    
    if (hasValidData) {
        m_buffer.append(data);
        emit newData(data);
    } else {
        emit errorOccurred("Invalid data received");
    }
    
    // If more data is available, schedule another read
    if (m_serialPort.bytesAvailable() > 0) {
        QTimer::singleShot(0, this, &SerialPortReader::handleReadyRead);
    }
}

void SerialPortReader::handleError(QSerialPort::SerialPortError error)
{
    if (error == QSerialPort::NoError) {
        return;
    }
    
    QString errorString = QString("Serial port error (%1): %2")
                         .arg(error)
                         .arg(m_serialPort.errorString());
    
    qCritical() << errorString;
    emit errorOccurred(errorString);
    
    // Handle critical errors by stopping the reader
    switch (error) {
        case QSerialPort::ResourceError:
        case QSerialPort::DeviceNotFoundError:
        case QSerialPort::PermissionError:
        case QSerialPort::UnknownError:
            qCritical() << "Critical error occurred, stopping reader";
            stop();
            break;
        case QSerialPort::ReadError:
        case QSerialPort::WriteError:
            qWarning() << "I/O error occurred, attempting to continue";
            // Clear buffers and try to recover
            m_serialPort.clear();
            clearBuffer();
            break;
        default:
            break;
    }
}

void SerialPortReader::handleTimeout()
{
    QMutexLocker locker(&m_mutex);
    
    qWarning() << "Read timeout occurred on port" << m_serialPort.portName();
    emit errorOccurred(QString("Read timeout on port %1").arg(m_serialPort.portName()));
    
    // Clear any partial data in buffer
    if (!m_buffer.isEmpty()) {
        qDebug() << "Clearing" << m_buffer.size() << "bytes from buffer due to timeout";
        clearBuffer();
    }
}

void SerialPortReader::clearBuffer()
{
    m_buffer.clear();
    m_buffer.squeeze();  // Release memory
}