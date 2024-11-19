MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent),
      serialReader(new SerialPortReader),
      startTime(QDateTime::currentDateTime().toMSecsSinceEpoch() / 1000.0),
      ecgPlot(new QCustomPlot),
      heartRatePlot(new QCustomPlot),
      heartRateLabel(new QCPItemText(heartRatePlot))  // Create the QCPItemText here
{
    ...
    // There is no need to add the item to heartRatePlot, it's automatically done in QCPItemText constructor
    ...
}
