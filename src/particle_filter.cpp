/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

// helper functions
// given two landmarks calculate their distance
double landmarkDist(LandmarkObs* a, LandmarkObs* b);
// give a particle and a observation in car coordinate, return transformed observation in map coordinate wrt the particle
LandmarkObs transform(LandmarkObs& obs, Particle p);
// return value of multivarate gaussian
double gaussian(double x[], double miu[], double sigma[]);

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO:
    // Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

    // set number of particles
    num_particles = 20;
    // sample from gaussian distribution of x, y and theta as init value
    double sigma_x = std[0];
    double sigma_y = std[1];
    double sigma_theta = std[2];
    default_random_engine gen;
    normal_distribution<double> dist_x {x, sigma_x};
    normal_distribution<double> dist_y {y, sigma_y};
    normal_distribution<double> dist_theta {theta, sigma_theta};
    // create list of particle
    for (int i = 0; i < num_particles; i++) {
        // sample x, y and theta
        double sample_x = dist_x(gen);
        double sample_y = dist_y(gen);
        double sample_theta = dist_theta(gen);
        // create particle {id, x, y, theta, weight
        Particle p {i, sample_x, sample_y, sample_theta, 1};
        // add to particle filter
        weights.push_back(1);
        particles.push_back(p);
    }
    // set init as true
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO:
    // Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
    double sigma_x = std_pos[0];
    double sigma_y = std_pos[1];
    double sigma_theta = std_pos[2];
    default_random_engine gen;
    normal_distribution<double> dist_x_noise {0, sigma_x};
    normal_distribution<double> dist_y_noise {0, sigma_y};
    normal_distribution<double> dist_theta_noise {0, sigma_theta};
    if (abs(yaw_rate) < 0.0000001) {
        for (int i = 0; i < num_particles; i++) {
            double x_noise = dist_x_noise(gen);
            double y_noise = dist_y_noise(gen);
            Particle& p = particles[i];
            p.x += delta_t * velocity * cos(p.theta) + x_noise;
            p.y += delta_t * velocity * sin(p.theta) + y_noise;
        }
    } else {
        for (int i = 0; i < num_particles; i++) {
            double x_noise = dist_x_noise(gen);
            double y_noise = dist_y_noise(gen);
            Particle& p = particles[i];
            p.x = p.x + (velocity / yaw_rate) * (sin(p.theta + yaw_rate * delta_t) - sin(p.theta)) + x_noise;
            p.y = p.y + (velocity / yaw_rate) * (cos(p.theta) - cos(p.theta + yaw_rate * delta_t)) + y_noise;
            p.theta = p.theta + yaw_rate * delta_t;
        }
    }
}

// predicted, landmarks within sensor range
// observations, transformed observations
void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO:
    // Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
    for (int i = 0; i < observations.size(); i++) {
        // set the first one in predicted as defaut association
        LandmarkObs* closest = &predicted[0];
        LandmarkObs* obs = &observations[i];
        double closest_dist = landmarkDist(closest, obs);
        // find the closest by loop through each landmarks
        for (int j = 0; j < predicted.size(); j++) {
            double dist = landmarkDist(&predicted[j], obs);
            if (dist < closest_dist) {
                closest = &predicted[j];
                closest_dist = dist;
            }
        }
        // assign the observation to the closest landmark
        obs -> id = closest -> id;
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO:
    // Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

    for (int i = 0; i < num_particles; i++) {
        Particle& p = particles[i]; // reference to the particle
        std::vector<LandmarkObs> tobs;
        // transform each observations
        for (int i = 0; i < observations.size(); i++) {
            tobs.push_back(transform(observations[i], p));
        }
        // find landmarks within sensor range
        std::vector<LandmarkObs> landmarks;
        for (auto const &landmark : map_landmarks.landmark_list) {
            LandmarkObs l {landmark.id_i, landmark.x_f, landmark.y_f};
            if (sqrt(pow(l.x - p.x, 2) + pow(l.y - p.y, 2)) <= sensor_range) {
                landmarks.push_back(l);
            }
        }
        // assign transform observations to closest landmark
        dataAssociation(landmarks, tobs);
        // calculating weights
        double weight = 1;
        for (auto const &tob : tobs) {
            double miu[] {map_landmarks.landmark_list[tob.id - 1].x_f, map_landmarks.landmark_list[tob.id - 1].y_f};
            double x[] {tob.x, tob.y};
            weight *= gaussian(x, miu, std_landmark);
        }
        p.weight = weight;
    }
    // put weights into filter
    for (auto const &p : particles) {
        weights.push_back(p.weight);
    }
}

void ParticleFilter::resample() {
	// TODO:
    // Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    std::discrete_distribution<double> d {weights.begin(), weights.end()};
    default_random_engine gen;
    vector<Particle> new_particles;
    for (int i = 0; i < num_particles; i++) {
        Particle p = particles[d(gen)];
        Particle new_p {p.id, p.x, p.y, p.theta, p.weight};
        new_particles.push_back(new_p);
    }
    particles = new_particles;
    weights.clear();
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

// return square value of distance between two landmarks
double landmarkDist(LandmarkObs *a, LandmarkObs *b) {
    return (a -> x - b -> x) * (a -> x - b -> x) + (a -> y - b -> y) * (a -> y - b -> y);
}

// given particle and a landmark in car coordinate
// return the transformed landmark in map coordinate
LandmarkObs transform(LandmarkObs& obs, Particle p) {
    LandmarkObs tobs;
    tobs.id = obs.id;
    tobs.x = obs.x * cos(p.theta) - obs.y * sin(p.theta) + p.x;
    tobs.y = obs.x * sin(p.theta) + obs.y * cos(p.theta) + p.y;
    return tobs;
}

double gaussian(double x[], double miu[], double sigma[]) {
    return exp(-(pow(x[0] - miu[0], 2) / (2 * pow(sigma[0], 2)) + pow(x[1] - miu[1], 2) / (2 * pow(sigma[1], 2)))) / (2 * M_PI * sigma[0] * sigma[1]);
}