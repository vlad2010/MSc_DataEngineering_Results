#include "mainwindow.h"
#include <QVBoxLayout>

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
    heartRatePlot->addGraph();

    connect(serialReader, &SerialPortReader::newECGData, this, &MainWindow::addECGData);
    connect(serialReader, &SerialPortReader::newHeartRateData, this, &MainWindow::addHeartRateData);

    serialReader->start();
}

MainWindow::~MainWindow()
{
    delete serialReader;
}

void MainWindow::addECGData(int data)
{
    double key = QDateTime::currentDateTime().toMSecsSinceEpoch() / 1000.0 - startTime;
    
    ecgData.append(data);
    ecgTime.append(key);

    if (ecgTime.size() > 360) {
        ecgTime.removeFirst();
        ecgData.removeFirst();
    }

    ecgPlot->graph(0)->setData(ecgTime, ecgData);
    ecgPlot->replot();
}

void MainWindow::addHeartRateData(unsigned int data)
{
    double key = QDateTime::currentDateTime().toMSecsSinceEpoch() / 1000.0 - startTime;
    
    heartRateData.append(data);
    heartRateTime.append(key);

    heartRatePlot->graph(0)->setData(heartRateTime, heartRateData);
    heartRatePlot->replot();
}
