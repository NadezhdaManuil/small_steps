
# include <iostream>
# include <future>
# include <iostream>
# include <windows.h>
# include <locale>
# include <codecvt>
# include <fstream>
# include <opencv2/core/core.hpp>
# include <opencv2/features2d/features2d.hpp>
# include <opencv2/highgui/highgui.hpp>
# include <opencv2/stitching/detail/util.hpp>
# include <opencv2/calib3d.hpp>
# include <opencv2/imgproc/imgproc.hpp>
# include <opencv2/core/persistence.hpp>
# include <filesystem>
# include <string>
# include <cstring>
# include <vector>
# include <stack>
# include <unordered_set>
# include <memory>
# include <thread>
# include <random>
# include <C:\Users\kokao\OneDrive\Рабочий стол\Vision_parklots\Vision_parklots\pugixml-1.12\src\pugixml.hpp>
# include <C:\Users\kokao\OneDrive\Рабочий стол\Vision_parklots\Vision_parklots\pugixml-1.12\src\pugixml.cpp>

using namespace std;
using namespace pugi;
using namespace cv;
namespace fs = filesystem;

vector<string> dirSearch(const string& str)
{
    vector<string> searchList;
    for (auto& p : fs::recursive_directory_iterator(str))
        if (!p.is_directory() && p.path().string().substr(p.path().string().length() - 3) != "jpg")
            searchList.push_back(p.path().string().erase(p.path().string().length() - 4));
    return searchList;
}




void prep(string mainPath, string dstPath) {
    int number_of_records = 0; //  for machine learning we dont need more data for a binary classifier
    int lowThreshold = 0;
    const int max_lowThreshold = 100;
    const int ratio = 5;
    const int kernel_size = 3;
    const char* window_name = "Edge Map";
    int MAX_FEATURES = 30;
    vector<string> ListPath = dirSearch(mainPath);
    fstream outputFile;
    outputFile.open(dstPath, std::ios::out);
    for (vector<string>::iterator it = ListPath.begin(); it != ListPath.end(); ++it) {
        if (number_of_records > 200000) break;
        string path_string = *it;
        string xmlfilepath = path_string + ".xml";
        const char* filepath = xmlfilepath.c_str();
        pugi::xml_document doc;
        pugi::xml_parse_result result = doc.load_file(filepath);
        for (pugi::xml_node space = doc.child("parking").child("space"); space; space = space.next_sibling("space"))
        {
            if (number_of_records > 200000) break;
            String occupied_label = space.attribute("occupied").value();
            String x = space.child("rotatedRect").child("center").attribute("x").value();
            String y = space.child("rotatedRect").child("center").attribute("y").value();
            String w = space.child("rotatedRect").child("size").attribute("w").value();
            String h = space.child("rotatedRect").child("size").attribute("h").value();
            String d = space.child("rotatedRect").child("angle").attribute("d").value();

            // to avvoid problems with different dimensions of out put pictures 
            // I use fixed w and h with the avrage values.

            if (!x.empty() && !y.empty() && !w.empty() && !h.empty() && !d.empty() && !occupied_label.empty()) {

                //stoi(occupied_label)
                try {
                    Mat3b image = imread(path_string + ".jpg");
                    RotatedRect rRect = RotatedRect(Point2f(stoi(x), stoi(y)), Size2f(50, 100), stoi(d));
                    Point2f vertices[4];
                    rRect.points(vertices);
                    for (int i = 0; i < 4; i++)
                        line(image, vertices[i], vertices[(i + 1) % 4], Scalar(0, 234, 255), 2);
                    //imshow("image", image);
                    //waitKey();
                    Rect brect = rRect.boundingRect2f();
                    rectangle(image, brect, Scalar(204, 255, 51));
                    Mat3b cropped = image(brect);
                    Ptr<Feature2D> orb = ORB::create(MAX_FEATURES);
                    std::vector<KeyPoint> keypoints;
                    Mat descriptors;
                    orb->detectAndCompute(cropped, Mat(), keypoints, descriptors);

                    /*
                    imshow("image", cropped);
                    drawKeypoints(cropped, keypoints, cropped);
                    imshow("image with keypoints", cropped);
                    waitKey();
                    */

                    for (size_t j = 0; j < keypoints.size(); ++j) {
                        outputFile << keypoints[j].pt.x << "," << keypoints[j].pt.y << ",";
                    }
                    if (keypoints.size() < MAX_FEATURES) {
                        for (size_t k = 0; k < (MAX_FEATURES - keypoints.size()); ++k) { //  to keep csv organized we fill the rest with 0
                            outputFile << 0 << "," << 0 << ",";                 // because we set max to 30 but we are not sure we get exactly
                        }
                    }
                    outputFile << stoi(occupied_label) << "\n"; //last column in csv is label
                    number_of_records++;


                }
                catch (Exception e) {
                    cout << e.err << endl;
                }
            }
        }
    }
    outputFile.close();
}
void makeCSVforOneImage() {
    int lowThreshold = 0;
    const int max_lowThreshold = 100;
    const int ratio = 5;
    const int kernel_size = 3;
    const char* window_name = "Edge Map";
    int MAX_FEATURES = 30;
    fstream outputFile;
    outputFile.open("projects_csvs\oneImageCSV.csv", std::ios::out);
    const char* filepath = "C:\\PKLot\\PKLot\\UFPR04\\Sunny\\2012-12-13\\2012-12-13_09_10_04.xml";
    pugi::xml_document doc;
    pugi::xml_parse_result result = doc.load_file(filepath);
    for (pugi::xml_node space = doc.child("parking").child("space"); space; space = space.next_sibling("space"))
    {
        cout << "here" << endl;
        String occupied_label = space.attribute("occupied").value();
        String x = space.child("rotatedRect").child("center").attribute("x").value();
        String y = space.child("rotatedRect").child("center").attribute("y").value();
        String w = space.child("rotatedRect").child("size").attribute("w").value();
        String h = space.child("rotatedRect").child("size").attribute("h").value();
        String d = space.child("rotatedRect").child("angle").attribute("d").value();

        // to avvoid problems with different dimensions of out put pictures 
        // I use fixed w and h with the avrage values.

        if (!x.empty() && !y.empty() && !w.empty() && !h.empty() && !d.empty() && !occupied_label.empty()) {

            //stoi(occupied_label)

            Mat3b image = imread("C:\\PKLot\\PKLot\\UFPR04\\Sunny\\2012-12-13\\2012-12-13_09_10_04.jpg");
            RotatedRect rRect = RotatedRect(Point2f(stoi(x), stoi(y)), Size2f(50, 100), stoi(d));
            Point2f vertices[4];
            rRect.points(vertices);
            for (int i = 0; i < 4; i++)
                line(image, vertices[i], vertices[(i + 1) % 4], Scalar(0, 234, 255), 2);
            //imshow("image", image);
            //waitKey();
            Rect brect = rRect.boundingRect2f();
            rectangle(image, brect, Scalar(204, 255, 51));
            Mat3b cropped = image(brect);
            Ptr<Feature2D> orb = ORB::create(MAX_FEATURES);
            std::vector<KeyPoint> keypoints;
            Mat descriptors;
            orb->detectAndCompute(cropped, Mat(), keypoints, descriptors);

            /*
            imshow("image", cropped);
            drawKeypoints(cropped, keypoints, cropped);
            imshow("image with keypoints", cropped);
            waitKey();
            */

            for (size_t j = 0; j < keypoints.size(); ++j) {
                outputFile << keypoints[j].pt.x << "," << keypoints[j].pt.y << ",";
            }
            if (keypoints.size() < MAX_FEATURES) {
                for (size_t k = 0; k < (MAX_FEATURES - keypoints.size()); ++k) { //  to keep csv organized we fill the rest with 0
                    outputFile << 0 << "," << 0 << ",";                 // because we set max to 30 but we are not sure we get exactly
                }
            }
            outputFile << stoi(occupied_label) << "\n"; //last column in csv is label

        }
    }

    outputFile.close();
}


void markPredictions() {
    int lowThreshold = 0;
    const int max_lowThreshold = 100;
    const int ratio = 5;
    const int kernel_size = 3;
    const char* window_name = "Edge Map";
    int MAX_FEATURES = 30;
    fstream outputFile;
    string l;
    vector<string> labels;
    int i = 0;
    fstream file("projects_csvs\preds_oneimage.csv");
    if (file.is_open())
    {
        while (getline(file, l))
        {
            labels.push_back(l);
        }
    }
    outputFile.open("projects_csvs\oneImageCSV.csv", std::ios::out);
    const char* filepath = "C:\\PKLot\\PKLot\\UFPR04\\Sunny\\2012-12-13\\2012-12-13_09_10_04.xml";
    string ImagePath = "C:\\PKLot\\PKLot\\UFPR04\\Sunny\\2012-12-13\\2012-12-13_09_10_04.jpg";
    Mat image = imread(ImagePath);
    pugi::xml_document doc;
    pugi::xml_parse_result result = doc.load_file(filepath);
    for (pugi::xml_node space = doc.child("parking").child("space"); space; space = space.next_sibling("space"))
    {

        String x = space.child("rotatedRect").child("center").attribute("x").value();
        String y = space.child("rotatedRect").child("center").attribute("y").value();
        String w = space.child("rotatedRect").child("size").attribute("w").value();
        String h = space.child("rotatedRect").child("size").attribute("h").value();
        String d = space.child("rotatedRect").child("angle").attribute("d").value();


        RotatedRect rRect = RotatedRect(Point2f(stoi(x), stoi(y)), Size2f(stoi(w), stoi(h)), stoi(d));
        Point2f vertices[4];
        rRect.points(vertices);
        if (stoi(labels.at(i)) == 0) {
            for (int i = 0; i < 4; i++)
                line(image, vertices[i], vertices[(i + 1) % 4], Scalar(100, 255, 20));
        }
        else {
            for (int i = 0; i < 4; i++)
                line(image, vertices[i], vertices[(i + 1) % 4], Scalar(199, 0, 255));

        }
        i++;
    }
    imshow("result of predictions: green = free ; red =occupied", image);
    waitKey();



}


void showResult() {

    makeCSVforOneImage();

    markPredictions();

}

int main()
{
    //preprocessing of data
    string pathOfDataFolder = "C:\\PKLot\\PKLot\\UFPR04";
    string pathOfdstCSV = "projects_csvs\output.csv";
    //prep(pathOfDataFolder, pathOfdstCSV);

    // after training the ml
    showResult();
}
