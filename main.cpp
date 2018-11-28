#include <fstream>
#include <iostream>
#include <vector>
#include <zeneural.hpp>
#include <random>

std::random_device rd;
std::mt19937 mt(rd());
std::uniform_real_distribution<double> randomNumbersGen(1.0, 1000000.0);

struct MNISTImage
{
    template <class InIt>
    MNISTImage(InIt start, InIt end) : bytes(start, end) {}

    template <class floatType>
    std::vector<floatType> getDataSet()
    {
        std::vector<floatType> temp;
        for (auto i : bytes)
        {
            temp.push_back(double((i < 180) ? -1.0 : 1.0));
        }
        return temp;
    }
    std::vector<uint8_t> bytes;
};

struct parseMNISTImages
{
    void moveToBuffer()
    {
        while (stream.good())
            buffer.push_back(stream.get());
    }
    void checkMagicNumber()
    {
        bool assert1 = buffer[2] == 8;
        bool assert2 = buffer[3] == 3;
        if (!(assert1 && assert2))
            throw 1;
    }
    void getInfo()
    {
        numberOfItems = 60000;
        height = (buffer[8] << 24) | (buffer[9] << 16) | (buffer[10] << 8) |
                 (buffer[11]);
        width = (buffer[12] << 24) | (buffer[13] << 16) | (buffer[14] << 8) |
                (buffer[15]);
    }
    void loadImages()
    {
        for (int i = 0; i < numberOfItems; i++)
        {
            int index = 12 + height * width * i;
            auto it1 = buffer.begin() + index;
            auto it2 = buffer.begin() + index + height * width;
            saves.push_back(MNISTImage(it1, it2));
        }
    }
    parseMNISTImages()
    {
        moveToBuffer();
        checkMagicNumber();
        std::cout << "[PROGRESS] Successfully Checked for the magic number \n";
        getInfo();
        std::cout << "[PROGRESS] Got Image Info. Images: " << numberOfItems
                  << ". Height: " << height << ". Width: " << width << "\n";
        loadImages();
        std::cout << "[PROGRESS] parsed the Images\n";
    }
    MNISTImage getImage(size_t index)
    {
        if (index < numberOfItems)
            return saves[index];
        throw 1;
    }
    int getDataSetInputSize() { return height * width; }
    void printImage(size_t index)
    {
        std::vector<double> image = getImage(index).getDataSet<double>();
        for (size_t i = 0; i < height; i++)
        {
            for (size_t j = 0; j < width; j++)
            {
                double tempByte = image[i * width + j];
                if (tempByte <= 0.5)
                {
                    std::cout << " ";
                }
                else
                {
                    std::cout << "█";
                }
            }
            std::cout << "\n";
        }
    }

  private:
    std::ifstream stream{"train-images.idx3-ubyte"};
    std::vector<MNISTImage> saves;
    std::vector<uint8_t> buffer;
    int numberOfItems = 0;
    int height = 0;
    int width = 0;
};

namespace UTIL
{
double tanh(double in)
{
    return std::tanh(in);
}
double tanhderivative(double in)
{
    return 1 - std::tanh(in) * std::tanh(in);
}
} // namespace UTIL

struct parseMNISTLabel
{
    void moveToBuffer()
    {
        while (stream.good())
            buffer.push_back(stream.get());
    }
    void checkMagicNumber()
    {
        bool assert1 = buffer[2] == 8;
        bool assert2 = buffer[3] == 1;
        if (!(assert1 && assert2))
            throw 1;
    }
    void getInfo()
    {
        numberOfItems =
            (buffer[4] << 24) | (buffer[5] << 16) | (buffer[6] << 8) | (buffer[7]);
    }
    void loadLabels() { buffer.erase(buffer.begin(), buffer.begin() + 8); }
    parseMNISTLabel()
    {
        moveToBuffer();
        checkMagicNumber();
        std::cout << "[PROGRESS] Successfully Checked for the magic number \n";
        getInfo();
        std::cout << "[PROGRESS] Got Label Info. Labels: " << numberOfItems << "\n";
        loadLabels();
        std::cout << "[PROGRESS] Loaded Labels \n";
    }
    unsigned int getLabel(size_t index)
    {
        if (index < numberOfItems)
            return buffer[index];
        throw 1;
    }
    template <class floatType>
    std::vector<floatType> getLabelAsVector(size_t index)
    {

        std::vector<floatType> temp;
        int result = getLabel(index);
        for (int i = 0; i < 10; i++)
        {
            if (i == result)
            {
                temp.push_back(1.0);
            }
            else
            {
                temp.push_back(-1.0);
            }
        }
        return temp;
    }

  private:
    std::ifstream stream{"train-labels.idx1-ubyte"};
    std::vector<uint8_t> buffer;
    int numberOfItems = 0;
};

class testNN
{
    parseMNISTImages images{};
    parseMNISTLabel labels{};
    ZNN::NeuralNetwork<double> nn;
    double sumOfErrors = 0.0;
    int batchSize = 1000;

  public:

    void setupNN()
    {
        nn.setInputLayerSize(images.getDataSetInputSize());
        nn.addHiddenLayer(400);
        nn.addHiddenLayer(2000);
        nn.setOutputLayerSize(10);
        nn.setLearningRate(0.009831415);
        nn.setNormalization(ZNN::Normalization<double>(UTIL::tanh, UTIL::tanhderivative));
    }

    testNN()
    {
        setupNN();
    }

    void evaluateProgress(int iteration)
    {
        std::cout << "iteration : " << iteration
                  << " . average error : " << sumOfErrors / 100 << "\n";
    }

    void trainNN(int iterations)
    {
        std::vector<double> expected;
        for (int i = 1; i < iterations; i++)
        {
            expected = labels.getLabelAsVector<double>(i % 60000);
            sumOfErrors +=
                nn.train(images.getImage(i % 60000).getDataSet<double>(), expected);
            if (i % 100 == 0)
            {
                evaluateProgress(i);
                sumOfErrors = 0;
            }
            if (i % batchSize == 0)
                testGuess(10, i);
        }
    }

    int getGuessOutput(std::vector<double> outputs)
    {
        int output = 0;
        double max = 0;
        for (size_t i = 0; i < outputs.size(); i++)
        {
            if (outputs[i] > max)
            {
                max = outputs[i];
                output = i;
            }
        }
        return output;
    }

    void testGuess(int amount, int startingPoint)
    {
        for (size_t i = startingPoint; i < startingPoint + amount; i++)
        {
            std::vector<double> temp = images.getImage(i % 60000).getDataSet<double>();
            auto outvec = nn.guess(temp);
            std::cout << "input: \n\n";
            for (size_t k = 0; k < temp.size(); k++)
            {
                if ((k) % 28 == 0)
                    std::cout << "\n";
                std::cout << ((temp[k] > 0.0) ? "##" : "  ");
            }
            std::cout << "\n\noutput: " << getGuessOutput (outvec) << "\n";
        };
        std::cout << "\n=====================================\n" << std::flush;
    }
};

int main()
{
    std::cout << std::fixed;
    testNN a;
    a.trainNN(10000000);
    a.testGuess(200, 1000);
}
