private:
    QCustomPlot *ecgPlot;
    QCustomPlot *heartRatePlot;
    SerialPortReader *serialReader;
    double startTime;
    double firstHeartRateTime;
    QVector<double> ecgTime;
    QVector<int> ecgData;
    QVector<double> heartRateTime;
    QVector<unsigned int> heartRateData;
