MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent),
      serialReader(new SerialPortReader),
      startTime(QDateTime::currentDateTime().toMSecsSinceEpoch() / 1000.0),
      ecgPlot(new QCustomPlot),
      heartRatePlot(new QCustomPlot)
{
    // Set up the serial port reader
    serialReader->setPortName("COM8");
    serialReader->setBaudRate(QSerialPort::Baud115200);
    connect(serialReader, &SerialPortReader::newEcgData, this, &MainWindow::addEcgData);
    connect(serialReader, &SerialPortReader::newHeartRateData, this, &MainWindow::addHeartRateData);
    
    // Set up the ECG plot
    ecgPlot->addGraph();
    ecgPlot->xAxis->setLabel("Time (s)");
    ecgPlot->yAxis->setLabel("ECG Signal");
    ecgPlot->yAxis->setRange(-500, 500);
    ecgPlot->plotLayout()->insertRow(0);
    ecgPlot->plotLayout()->addElement(0, 0, new QCPTextElement(ecgPlot, "ECG Plot", QFont("sans", 12, QFont::Bold)));

    // Set up the heart rate plot
    heartRatePlot->addGraph();
    heartRatePlot->xAxis->setLabel("Time (s)");
    heartRatePlot->yAxis->setLabel("Heart Rate (bpm)");
    heartRatePlot->graph(0)->setLineStyle(QCPGraph::lsLine);
    heartRatePlot->graph(0)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssDisc, 5)); // Added this line
    heartRatePlot->plotLayout()->insertRow(0);
    heartRatePlot->plotLayout()->addElement(0, 0, new QCPTextElement(heartRatePlot, "Heart Rate Plot", QFont("sans", 12, QFont::Bold)));
    
    // Set up the layouts
    QVBoxLayout *vLayout = new QVBoxLayout;
    vLayout->addWidget(ecgPlot);
    vLayout->addWidget(heartRatePlot);
    QWidget *windowContent = new QWidget;
    windowContent->setLayout(vLayout);
    setCentralWidget(windowContent);
}
