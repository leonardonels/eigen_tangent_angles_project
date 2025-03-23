#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <Eigen/Dense>
#include <vector>
#include <numeric>
#include <algorithm>

// LQR **********************************************************
Eigen::Vector2d subtract(const Eigen::Vector2d &a, const Eigen::Vector2d &b) {
    return {a[0] - b[0], a[1] - b[1]};
}

Eigen::Vector2d normalize(const Eigen::Vector2d &p) {
    double len = std::sqrt(p[0] * p[0] + p[1] * p[1]);
    if (len == 0) return {0, 0};  // Prevent division by zero
    return {p[0] / len, p[1] / len};
}

double distance(const Eigen::Vector2d &a, const Eigen::Vector2d &b) {
    double dx = a[0] - b[0];
    double dy = a[1] - b[1];
    return std::sqrt(dx * dx + dy * dy);
}

std::vector<Eigen::Vector2d> convert_to_vector2d(const Eigen::VectorXd& x, const Eigen::VectorXd& y) {
    std::vector<Eigen::Vector2d> points;
    for (int i = 0; i < x.size(); ++i) {
        points.push_back(Eigen::Vector2d(x(i), y(i)));  // Pair x and y into Eigen::Vector2d
    }
    return points;
}

std::vector<double> get_tangent_angles(std::vector<Eigen::Vector2d> points) {
    std::vector<double> tangent_angles(points.size());
    
    if (points.size() >= 2) {
        // First point: forward difference.
        Eigen::Vector2d diff = subtract(points[1], points[0]);
        Eigen::Vector2d tanVec = normalize(diff);
        tangent_angles[0] = std::atan2(tanVec[1], tanVec[0]);

        // Last point: backward difference.
        diff = subtract(points.back(), points[points.size() - 2]);
        tanVec = normalize(diff);
        tangent_angles.back() = std::atan2(tanVec[1], tanVec[0]);
    }

     for (size_t i = 1; i < points.size() - 1; ++i) {
        double d1 = distance(points[i], points[i - 1]);
        double d2 = distance(points[i + 1], points[i]);
        double ds = d1 + d2;  // Total distance over the two segments

        if (ds == 0) {
            tangent_angles[i] = 0;  // Fallback if points coincide
        } else {
            // Compute the central difference divided by the total arc length.
            Eigen::Vector2d diff = {
                (points[i + 1][0] - points[i - 1][0]) / ds,
                (points[i + 1][1] - points[i - 1][1]) / ds
            };
            Eigen::Vector2d tanVec = normalize(diff);
            tangent_angles[i] = std::atan2(tanVec[1], tanVec[0]);
        }
    }

    return tangent_angles;
}

// CSV **********************************************************
template <typename T>
std::pair<Eigen::VectorXd, Eigen::VectorXd> read_csv(const char* input_filename) {
    std::ifstream input(input_filename);
    if (!input) {
        std::cerr << "Error opening input file: " << input_filename << "\n";
        return {};
    }

    std::string line;
    std::vector<T> x, y;

    std::getline(input, line); // Skip the header

    while (std::getline(input, line)) {
        std::istringstream ss(line);
        std::string token;
        std::vector<T> values;
        
        // Read comma-separated values
        while (std::getline(ss, token, ',')) {
            try {
                // Try to convert each token to a double
                values.push_back(std::stod(token));
            } catch (const std::invalid_argument& e) {
                std::cerr << "Error: Invalid data found in file: " << token << "\n";
                continue; // Skip this line if it contains invalid data
            } catch (const std::out_of_range& e) {
                std::cerr << "Error: Data out of range in file: " << token << "\n";
                continue; // Skip this line if data is out of range
            }
        }

        if (values.size() >= 2) {
            T csv_x = values[0];
            T csv_y = values[1];
            x.push_back(csv_x);
            y.push_back(csv_y);
        }
    }

    Eigen::VectorXd X(x.size());
    Eigen::VectorXd Y(y.size());

    for (size_t i = 0; i < x.size(); ++i) {
        X(i) = x[i];
        Y(i) = y[i];
    }

    return std::make_pair(X, Y);
}

// EIGEN ********************************************************
void computeSplineCoefficients(const Eigen::VectorXd& x, const Eigen::VectorXd& y, 
                                Eigen::VectorXd& a, Eigen::VectorXd& b, 
                                Eigen::VectorXd& c, Eigen::VectorXd& d) {
    int n = x.size();
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(n, n);  // System matrix
    Eigen::VectorXd B = Eigen::VectorXd::Zero(n);     // Right-hand side vector
    Eigen::VectorXd C = Eigen::VectorXd::Zero(n);     // Solution vector for second derivatives

    // Step 1: Set up the system of equations
    // Boundary conditions (for closed spline)
    A(0, 0) = 1;  // For the first point, set the second derivative to zero
    A(n - 1, n - 1) = 1;  // For the last point, set the second derivative to zero

    // Create the system for the second derivatives
    for (int i = 1; i < n - 1; ++i) {
        double h1 = x[i] - x[i - 1];
        double h2 = x[i + 1] - x[i];
        
        A(i, i - 1) = h1;
        A(i, i) = 2 * (h1 + h2);
        A(i, i + 1) = h2;
        
        B(i) = 3 * ((y[i + 1] - y[i]) / h2 - (y[i] - y[i - 1]) / h1);
    }

    // Step 2: Solve for second derivatives (C)
    C = A.colPivHouseholderQr().solve(B);

    // Step 3: Calculate the coefficients a, b, c, and d for each spline segment
    a = y;  // a_i = y_i

    for (int i = 0; i < n - 1; ++i) {
        double h = x[i + 1] - x[i];
        c[i] = C[i];
        d[i] = (C[i + 1] - C[i]) / (3 * h);
        b[i] = (y[i + 1] - y[i]) / h - h * (C[i + 1] + 2 * C[i]) / 3;
    }

    // For the last point, apply the boundary condition for the closed spline
    double h = x[n - 1] - x[n - 2];
    c[n - 1] = C[n - 1];
    d[n - 1] = (C[0] - C[n - 1]) / (3 * h);
    b[n - 1] = (y[0] - y[n - 1]) / h - h * (C[0] + 2 * C[n - 1]) / 3;
}

double firstDerivative(double x_query, const Eigen::VectorXd& x, const Eigen::VectorXd& a,
                       const Eigen::VectorXd& b, const Eigen::VectorXd& c, const Eigen::VectorXd& d) {
    int n = x.size();
    
    for (int i = 0; i < n - 1; ++i) {
        if (x_query >= x[i] && x_query <= x[i + 1]) {
            double dx = x_query - x[i];
            return b[i] + 2 * c[i] * dx + 3 * d[i] * dx * dx;
        }
    }

    double dx = x_query - x[n-1];
    return b[n-1] + 2 * c[n-1] * dx + 3 * d[n-1] * dx * dx;
}

double secondDerivative(double x_query, const Eigen::VectorXd& x, const Eigen::VectorXd& a,
                        const Eigen::VectorXd& b, const Eigen::VectorXd& c, const Eigen::VectorXd& d) {
    int n = x.size();
    
    for (int i = 0; i < n - 1; ++i) {
        if (x_query >= x[i] && x_query <= x[i + 1]) {
            double dx = x_query - x[i];
            return 2 * c[i] + 6 * d[i] * dx;
        }
    }

    double dx = x_query - x[n-1];
    return 2 * c[n-1] + 6 * d[n-1] * dx;
}

// UTILS ********************************************************
template<typename T>
std::vector<T> elem_wise_vector_diff(const std::vector<T>& vec1, const std::vector<T>& vec2) {
    if (vec1.size() != vec2.size()) {
        std::cerr << "Vectors have different sizes!" << std::endl;
        return {};
    }

    std::vector<T> absolute_differences(vec1.size());

    std::transform(vec1.begin(), vec1.end(), vec2.begin(), absolute_differences.begin(),
                   [](T a, T b) { return std::abs(a - b); });

    return absolute_differences;
}

template<typename T>
double vector_mean(const std::vector<T>& v){
    double sum = std::accumulate(v.begin(), v.end(), 0.0);
    return sum / v.size();
}


// main *********************************************************
int main(int argc, char* argv[]) {

    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <input file>\n";
        return 1;
    }

    // CSV
    auto data = read_csv<double>(argv[1]);
    if (data.first.size() == 0 || data.second.size() == 0) {
        std::cerr << "Error: No valid data found in the file.\n";
        return 1;
    }
    
    Eigen::VectorXd x = data.first;
    Eigen::VectorXd y = data.second;
    
    // LQR
    std::vector<Eigen::Vector2d> points = convert_to_vector2d(x, y);
    std::vector<double> points_tangents = get_tangent_angles(points);
    
    // EIGEN
    int n = x.size();
    Eigen::VectorXd a(n), b(n), c(n), d(n);
    computeSplineCoefficients(x, y, a, b, c, d);

    std::vector<double> tangent_angles;
    std::cout << "First and Second Derivatives at Data Points:" << std::endl;
    for (int i = 0; i < n - 1; ++i) {
        double x_query = x[i];
        double first_deriv = firstDerivative(x_query, x, a, b, c, d);
        double second_deriv = secondDerivative(x_query, x, a, b, c, d);

        double tangent_angle = std::atan(first_deriv);
        tangent_angles.push_back(tangent_angle);

        std::cout << "Point (" << x_query << ", " << y[i] << ") -> "
                  << "First Derivative (Slope): " << first_deriv
                  << ", Second Derivative (Curvature): " << second_deriv 
                  << ", Tangent Angle: " << tangent_angle 
                  << ", LQR Tangent Angle: " << points_tangents[i] << std::endl;
    }

    double x_query = x[n - 1];
    double first_deriv = firstDerivative(x_query, x, a, b, c, d);
    double second_deriv = secondDerivative(x_query, x, a, b, c, d);

    double tangent_angle = std::atan(first_deriv);
    tangent_angles.push_back(tangent_angle);

    std::cout << "Point (" << x_query << ", " << y[n - 1] << ") -> "
              << "First Derivative (Slope): " << first_deriv
              << ", Second Derivative (Curvature): " << second_deriv 
              << ", Tangent Angle: " << tangent_angle 
              << ", LQR Tangent Angle: " << points_tangents[n - 1] << std::endl;

    std::vector<double> absolute_differences = elem_wise_vector_diff<double>(tangent_angles, points_tangents);
    if (absolute_differences.size() == 0) {
        std::cerr << "Error: Absolute difference computation failed.\n";
        return 1;
    }
    
    for(int i = 0; i < n - 1; ++i){
        std::cout << "Error for point: "<< i
        << " -> " << absolute_differences[i] << std::endl;
    }
    std::cout << "Error for point: "<< n - 1
    << " -> " << absolute_differences[n - 1] << std::endl;

    double mean_error = vector_mean<double>(absolute_differences);
    
    std::cout << "Absolute Mean Error: " << mean_error << std::endl;

    return 0;
}
