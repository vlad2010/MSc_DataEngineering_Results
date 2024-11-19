MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ecgPlot(new QCustomPlot(this)),
    heartRatePlot(new QCustomPlot(this)),
    startTime(QDateTime::currentDateTime().toMSecsSinceEpoch() / 1000.0),
    serialReader(new SerialPortReader(this))
{
    // ... Layout setup ...

    connect(serialReader, &SerialPortReader::newECGData, this, &MainWindow::addECGData);
    connect(serialReader, &SerialPortReader::newHeartRateData, this, &MainWindow::addHeartRateData);

    serialReader->start();
}
