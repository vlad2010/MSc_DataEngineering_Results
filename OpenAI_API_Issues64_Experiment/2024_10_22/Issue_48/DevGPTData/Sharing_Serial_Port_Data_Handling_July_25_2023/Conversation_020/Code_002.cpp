#include <QVBoxLayout>
#include <QTimer>
#include "mainwindow.h"
#include "qcp/qcp.h"  // Add this line for QCPItemText

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent),
      serialReader(new SerialPortReader),
      startTime(QDateTime::currentDateTime().toMSecsSinceEpoch() / 1000.0),
      ecgPlot(new QCustomPlot),
      heartRatePlot(new QCustomPlot),
      heartRateLabel(new QCPItemText(heartRatePlot))  // Add this line to create the QCPItemText object
{
    QVBoxLayout *mainLayout = new QVBoxLayout;
    QWidget *centralWidget = new QWidget;

    // Configure ecgPlot...
    
    // Configure heartRatePlot...

    mainLayout->addWidget(ecgPlot);
    mainLayout->addWidget(heartRatePlot);

    centralWidget->setLayout(mainLayout);
    setCentralWidget(centralWidget);

    // Connect SerialPortReader signals to MainWindow slots
    connect(serialReader, &SerialPortReader::newEcgData, this, &MainWindow::addEcgData);
    connect(serialReader, &SerialPortReader::newHeartRateData, this, &MainWindow::addHeartRateData);
}

MainWindow::~MainWindow()
{
    delete serialReader;
}

void MainWindow::addEcgData(int data)
{
    // Add ECG data to the plot...
}

void MainWindow::addHeartRateData(unsigned int data)
{
    double key = QDateTime::currentDateTime().toMSecsSinceEpoch() / 1000.0 - startTime;

    heartRateData.append(data);
    heartRateTime.append(key);

    heartRatePlot->graph(0)->setData(heartRateTime, heartRateData);
    heartRatePlot->xAxis->setRange(*std::min_element(heartRateTime.constBegin(), heartRateTime.constEnd()),
                                   *std::max_element(heartRateTime.constBegin(), heartRateTime.constEnd()) + 1);
    heartRatePlot->yAxis->setRange(*std::min_element(heartRateData.constBegin(), heartRateData.constEnd()) - 1,
                                   *std::max_element(heartRateData.constBegin(), heartRateData.constEnd()) + 1);

    // Update the label
    heartRateLabel->position->setType(QCPItemPosition::ptPlotCoords); 
    heartRateLabel->position->setCoords(key, data); 
    heartRateLabel->setText(QString::number(data));
    heartRateLabel->setFont(QFont(font().family(), 8));
    heartRateLabel->setColor(Qt::black);
    heartRateLabel->setPadding(QMargins(1, 1, 1, 1));

    heartRatePlot->replot();
}
