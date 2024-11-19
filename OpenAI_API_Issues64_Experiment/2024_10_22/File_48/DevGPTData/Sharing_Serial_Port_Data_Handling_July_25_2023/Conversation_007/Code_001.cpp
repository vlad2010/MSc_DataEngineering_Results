MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ecgPlot(new QCustomPlot(this)),
    heartRatePlot(new QCustomPlot(this)),
    startTime(QDateTime::currentDateTime().toMSecsSinceEpoch() / 1000.0),
    serialReader(new SerialPortReader(this))
{
    QVBoxLayout *layout = new QVBoxLayout;
    layout->addWidget(ecgPlot);
    layout->addWidget(heartRatePlot);

    QWidget *mainWidget = new QWidget(this);
    mainWidget->setLayout(layout);

    setCentralWidget(mainWidget);

    ecgPlot->addGraph();
    ecgPlot->xAxis->setRange(0, 10);
    ecgPlot->yAxis->setRange(-600, 600);

    heartRatePlot->addGraph();
    heartRatePlot->xAxis->setRange(0, 10);
    heartRatePlot->yAxis->setRange(0, 200); // Assuming the heart rate will be between 0 and 200 bpm

    connect(serialReader, &SerialPortReader::newECGData, this, &MainWindow::addECGData);
    connect(serialReader, &SerialPortReader::newHeartRateData, this, &MainWindow::addHeartRateData);

    serialReader->start();
}
