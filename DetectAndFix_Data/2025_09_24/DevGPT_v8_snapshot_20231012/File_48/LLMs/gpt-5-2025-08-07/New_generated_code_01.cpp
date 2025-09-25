// serialportreader_secure.cpp
#include <QObject>
#include <QSerialPort>
#include <QSerialPortInfo>
#include <QTimer>
#include <QByteArray>
#include <QDebug>
#include <QCoreApplication>
#include <QtGlobal>

class SerialPortReader : public QObject
{
    Q_OBJECT
public:
    explicit SerialPortReader(QObject *parent = nullptr)
        : QObject(parent)
    {
        // Explicit, known-good serial settings
        m_serialPort.setBaudRate(QSerialPort::Baud115200);
        m_serialPort.setDataBits(QSerialPort::Data8);
        m_serialPort.setParity(QSerialPort::NoParity);
        m_serialPort.setStopBits(QSerialPort::OneStop);
        m_serialPort.setFlowControl(QSerialPort::NoFlowControl);

        // Mitigation for CWE-770/CWE-400: apply backpressure with a bounded read buffer
        m_serialPort.setReadBufferSize(kMaxReadBufferBytes);

        connect(&m_serialPort, &QSerialPort::readyRead,
                this, &SerialPortReader::handleReadyRead);

        // Mitigation for CWE-703: handle all errors and recover safely
        connect(&m_serialPort, &QSerialPort::errorOccurred,
                this, &SerialPortReader::handlePortError);

        // Controlled reconnect/backoff
        m_reconnectTimer.setSingleShot(true);
        connect(&m_reconnectTimer, &QTimer::timeout, this, &SerialPortReader::attemptOpen);
    }

    ~SerialPortReader() override
    {
        if (m_serialPort.isOpen()) {
            m_serialPort.close();
        }
    }

    void start()
    {
        // Attempt to open immediately
        attemptOpen();
    }

signals:
    void newData(const QByteArray &data);

private slots:
    void handleReadyRead()
    {
        // Process input in bounded chunks to avoid large intermediate allocations.
        // This mitigates memory spikes and reduces risk if upstream consumers are slow.
        int iterations = 0;
        while (m_serialPort.bytesAvailable() > 0 && iterations < kMaxReadIterationsPerEvent) {
            const qint64 toRead = qMin<qint64>(kChunkSizeBytes, m_serialPort.bytesAvailable());
            QByteArray chunk = m_serialPort.read(toRead);
            if (chunk.isEmpty()) {
                // No more data or read error; break to avoid tight loop
                break;
            }
            emit newData(chunk);
            ++iterations;
        }
        // If data continues to arrive quickly, Qt will emit readyRead again.
    }

    void handlePortError(QSerialPort::SerialPortError error)
    {
        if (error == QSerialPort::NoError) {
            return;
        }

        // Minimal structured logging (mitigate CWE-209: do not dump raw system strings in production)
        qWarning() << "[Serial]" << "error code:" << static_cast<int>(error);

        // Close and back off on severe errors
        switch (error) {
            case QSerialPort::DeviceNotFoundError:
            case QSerialPort::PermissionError:
            case QSerialPort::OpenError:
            case QSerialPort::ResourceError:
            case QSerialPort::UnknownError:
                safeCloseAndBackoff();
                break;

            case QSerialPort::ReadError:
            case QSerialPort::WriteError:
            case QSerialPort::TimeoutError:
            case QSerialPort::ParityError:
            case QSerialPort::FramingError:
            case QSerialPort::BreakConditionError:
            default:
                // For transient I/O errors, keep the port open if possible.
                // If the port is no longer functional, the subsequent read/write will fail and trigger another error.
                break;
        }
    }

    void attemptOpen()
    {
        if (m_serialPort.isOpen()) {
            return;
        }

        // Select a safe, validated port from available ports
        const QString port = selectSafePort();
        if (port.isEmpty()) {
            // No acceptable port available; retry later with backoff
            scheduleReconnect();
            return;
        }

        if (m_serialPort.portName() != port) {
            m_serialPort.setPortName(port);
        }

        if (m_serialPort.open(QIODevice::ReadOnly)) {
            qInfo() << "[Serial]" << "Opened" << m_serialPort.portName()
                    << "at" << m_serialPort.baudRate() << "bps";
            m_backoffMs = kInitialBackoffMs; // reset backoff on success
        } else {
            // Minimal, safe log. Avoid dumping full errorString in production logs.
            qWarning() << "[Serial]" << "Open failed for" << port
                       << "code:" << static_cast<int>(m_serialPort.error());
            scheduleReconnect();
        }
    }

private:
    void safeCloseAndBackoff()
    {
        if (m_serialPort.isOpen()) {
            m_serialPort.close();
        }
        scheduleReconnect();
    }

    void scheduleReconnect()
    {
        m_backoffMs = qMin(m_backoffMs * 2, kMaxBackoffMs); // exponential backoff with cap
        if (!m_reconnectTimer.isActive()) {
            m_reconnectTimer.start(m_backoffMs);
        }
    }

    // Choose a port safely:
    // - Only return ports that are currently available.
    // - Optionally apply an allowlist (VID/PID/serial).
    // - Optionally accept a configured port (env/config) but validate it exists.
    QString selectSafePort() const
    {
        // Optional allowlist by VID/PID (fill with your known device IDs)
        static const QList<QPair<quint16, quint16>> kAllowedVidPid = {
            // Example: {0x2341, 0x0043}, // Arduino Uno
            // {0x10C4, 0xEA60}  // Silicon Labs CP210x
        };

        const QList<QSerialPortInfo> ports = QSerialPortInfo::availablePorts();

        // 1) If ENV SERIAL_PORT is set, only accept it if it's present in availablePorts
        const QByteArray envPort = qgetenv("SERIAL_PORT");
        if (!envPort.isEmpty()) {
            const QString wanted = QString::fromLocal8Bit(envPort);
            for (const QSerialPortInfo &pi : ports) {
                if (pi.portName() == wanted && (kAllowedVidPid.isEmpty() ||
                        kAllowedVidPid.contains({pi.vendorIdentifier(), pi.productIdentifier()}))) {
                    return pi.portName();
                }
            }
            // If env is invalid, fall through to other selection
        }

        // 2) If "COM8" is desired, only use it if present and allowed
        const QString hardcoded = QStringLiteral("COM8");
        for (const QSerialPortInfo &pi : ports) {
            if (pi.portName() == hardcoded && (kAllowedVidPid.isEmpty() ||
                    kAllowedVidPid.contains({pi.vendorIdentifier(), pi.productIdentifier()}))) {
                return pi.portName();
            }
        }

        // 3) Otherwise, pick the first allowlisted device, if any
        if (!kAllowedVidPid.isEmpty()) {
            for (const QSerialPortInfo &pi : ports) {
                if (kAllowedVidPid.contains({pi.vendorIdentifier(), pi.productIdentifier()})) {
                    return pi.portName();
                }
            }
            return QString();
        }

        // 4) As a last resort, pick the first available port (least preferred)
        if (!ports.isEmpty()) {
            return ports.first().portName();
        }

        return QString();
    }

private:
    // Tunables
    static constexpr qint64 kMaxReadBufferBytes = 64 * 1024;  // Backpressure limit (64 KiB)
    static constexpr qint64 kChunkSizeBytes     = 4 * 1024;   // Read chunk size (4 KiB)
    static constexpr int    kInitialBackoffMs   = 500;
    static constexpr int    kMaxBackoffMs       = 8000;
    static constexpr int    kMaxReadIterationsPerEvent = 64;  // Prevent long monopolization of the event loop

    QSerialPort m_serialPort;
    QTimer m_reconnectTimer;
    int m_backoffMs = kInitialBackoffMs;
};

// If you are using a separate moc step, ensure this file is processed by moc or move the class to a header.
#include "serialportreader_secure.moc"