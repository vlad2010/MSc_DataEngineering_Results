void SerialPortReader::handleReadyRead()
{
    m_readData.append(m_serialPort.readAll());

    while (m_readData.contains('\n')) {
        QByteArray line = m_readData.left(m_readData.indexOf('\n'));
        m_readData.remove(0, line.size() + 1);

        line = line.trimmed();

        if (line.contains(',')) {
            QList<QByteArray> parts = line.split(',');
            emit newECGData(parts[0].toInt());
            emit newHeartRateData(parts[1].toUInt());
        } else {
            emit newECGData(line.toInt());
        }
    }
}
