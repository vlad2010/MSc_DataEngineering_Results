signals:
    void newECGData(int ecgData);
    void newHeartRateData(unsigned int heartRate);

private:
    QByteArray m_readData;
