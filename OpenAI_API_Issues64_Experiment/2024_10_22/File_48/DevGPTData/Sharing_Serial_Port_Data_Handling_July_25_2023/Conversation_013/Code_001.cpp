MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent),
      serialReader(new SerialPortReader),
      startTime(QDateTime::currentDateTime().toMSecsSinceEpoch() / 1000.0),
      ecgPlot(new QCustomPlot),
      heartRatePlot(new QCustomPlot),
      latestHeartRateLabel(new QLabel(this))
{
    // Set up the layouts, plots and connections...

    // Place the label. Adjust the position and size as needed.
    latestHeartRateLabel->setGeometry(QRect(50, 50, 200, 50));
}
