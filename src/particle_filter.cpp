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

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    // Set the number of particles. Initialize all particles to first position (based on estimates of 
    //   x, y, theta and their uncertainties from GPS) and all weights to 1. 
    // Add random Gaussian noise to each particle.
    // NOTE: Consult particle_filter.h for more information about this method (and others in this file).

    num_particles = 100;

    default_random_engine gen;

    double std_x, std_y, std_theta; // Standard deviations for x, y, and theta

    // Set standard deviations for x, y, and theta.
    std_x = std[0];
    std_y = std[1];
    std_theta = std[2];

    // Create a normal (Gaussian) distribution for x.
    normal_distribution<double> dist_x(x, std_x);
    // Create normal distributions for y.
    normal_distribution<double> dist_y(y, std_y);
    // Create normal distributions for theta.
    normal_distribution<double> dist_theta(theta, std_theta);

    Particle p;
    for (size_t i = 0; i < num_particles; i++)
    {
        p.id = i;
        p.x = dist_x(gen);
        p.y = dist_y(gen);
        p.theta = dist_theta(gen);
        p.weight = 1.0;

        particles.push_back(p);
        weights.push_back(1.0);
    }

    // Set initialized flag as true
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    // TODO: Add measurements to each particle and add random Gaussian noise.
    // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
    //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
    //  http://www.cplusplus.com/reference/random/default_random_engine/

    default_random_engine gen;
    weights.clear();

    // Set standard deviations for x, y, and theta.
    double std_x, std_y, std_theta; // Standard deviations for x, y, and theta
    std_x = std_pos[0];
    std_y = std_pos[1];
    std_theta = std_pos[2];

    double x_final = 0.0;
    double y_final = 0.0;
    double theta_final = 0.0;

    for (Particle & tempParticle : particles)
    {
        x_final = tempParticle.x;
        y_final = tempParticle.y;
        theta_final = tempParticle.theta;

        //cout << "particle before update, x: " << x_final << " y: " << y_final << " theta: " << theta_final << endl;

        if (yaw_rate == 0)
        {
            x_final += velocity * delta_t * cos(theta_final);
            y_final += velocity * delta_t * sin(theta_final);
        }
        else
        {
            x_final += velocity * (sin(theta_final + yaw_rate * delta_t) - sin(theta_final)) / yaw_rate;
            y_final += velocity * (cos(theta_final) - cos(theta_final + yaw_rate * delta_t)) / yaw_rate;
            theta_final += yaw_rate * delta_t;
        }
        //cout << "particle after update, x: " << x_final << " y: " << y_final << " theta: " << theta_final << endl;

        // Create a normal (Gaussian) distribution for x.
        normal_distribution<double> dist_x(x_final, std_x);
        // Create normal distributions for y.
        normal_distribution<double> dist_y(y_final, std_y);
        // Create normal distributions for theta.
        normal_distribution<double> dist_theta(theta_final, std_theta);

        tempParticle.x = dist_x(gen);
        tempParticle.y = dist_y(gen);
        tempParticle.theta = dist_theta(gen);

        //cout << "particle after adding noise, x: " << tempParticle.x << " y: " << tempParticle.y << " theta: " << tempParticle.theta << endl;

    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
    // TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
    //   observed measurement to this particular landmark.
    // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
    //   implement this method and use it as a helper during the updateWeights phase.

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
    const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
    // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
    //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
    //   according to the MAP'S coordinate system. You will need to transform between the two systems.
    //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
    //   The following is a good resource for the theory:
    //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
    //   and the following is a good resource for the actual equation to implement (look at equation 
    //   3.33
    //   http://planning.cs.uiuc.edu/node99.html

    // Transform the observations in the car coordinate (x_c and y_c) into map coordinate (x_m and y_m)
    vector<LandmarkObs> transformed_observations;
    size_t num_observations = observations.size();

    for (Particle & tempParticle : particles)
    {
        transformed_observations.clear();
        tempParticle.reset();

        for (const LandmarkObs & tempObs : observations)
        {
            ComputeHomogenousTransformation(tempParticle, tempObs, transformed_observations);
        }

        for (LandmarkObs & transformed_obs : transformed_observations)
        {
            double shortest_dist = sensor_range;
            Map::single_landmark_s closest_landmark;
            for (const Map::single_landmark_s & landmark : map_landmarks.landmark_list)
            {
                double euclidean_dist = dist(landmark.x_f, landmark.y_f, transformed_obs.x, transformed_obs.y);
                if (euclidean_dist < shortest_dist)
                {
                    //cout << "For transformed observation x: " << transformed_obs.x << " y: " << transformed_obs.y << " nearby landmark id: " << landmark.id_i << endl;
                    shortest_dist = euclidean_dist;
                    closest_landmark = landmark;
                }
            }

            if (closest_landmark.id_i != 0)
            {
                //cout << "For transformed observation x: " << transformed_obs.x << " y: " << transformed_obs.y << " closest landmark id: " << closest_landmark.id_i << endl;
                //cout << "Particle weight: " << tempParticle.weight << " ";
                tempParticle.weight *= CalculateParticleWeight(std_landmark, transformed_obs, closest_landmark);
                //cout << " updated weight: " << tempParticle.weight << endl;
                tempParticle.associations.push_back(closest_landmark.id_i);
                tempParticle.sense_x.push_back(closest_landmark.x_f);
                tempParticle.sense_y.push_back(closest_landmark.y_f);
                //tempParticle = SetAssociations(tempParticle, tempParticle.associations, tempParticle.sense_x, tempParticle.sense_y);
            }
        }

        weights.push_back(tempParticle.weight);
    }
}

double ParticleFilter::CalculateParticleWeight(double * std_landmark, LandmarkObs & transformed_obs, Map::single_landmark_s &closest_landmark)
{
    double normalization_term = 1 / (2 * M_PI * std_landmark[0] * std_landmark[1]);
    double term_x = pow(transformed_obs.x - closest_landmark.x_f, 2) / (2 * std_landmark[0] * std_landmark[0]);
    double term_y = pow(transformed_obs.y - closest_landmark.y_f, 2) / (2 * std_landmark[1] * std_landmark[1]);
    return normalization_term * exp(-(term_x + term_y));
}

void ParticleFilter::ComputeHomogenousTransformation(Particle & tempParticle, const LandmarkObs & tempObs, std::vector<LandmarkObs> &transformed_observations)
{
    LandmarkObs transformed_obs;
    // x_m=x_p​+(cos_theta*x_c​)−(sin_theta*y_c​)
    transformed_obs.x = tempParticle.x + cos(tempParticle.theta)*tempObs.x - sin(tempParticle.theta)*tempObs.y;
    // y_m=y_p​+(sin_theta*x_c​)+(cos_theta*y_c​)
    transformed_obs.y = tempParticle.y + sin(tempParticle.theta)*tempObs.x + cos(tempParticle.theta)*tempObs.y;
    transformed_observations.push_back(transformed_obs);
}

void ParticleFilter::resample() {
    // TODO: Resample particles with replacement with probability proportional to their weight. 
    // NOTE: You may find std::discrete_distribution helpful here.
    //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

    std::default_random_engine generator;
    // Using discrete_distribution's range constructor
    // http://www.cplusplus.com/reference/random/discrete_distribution/discrete_distribution/
    std::discrete_distribution<int> distribution(weights.begin(), weights.end());

    std::vector<Particle> temp_particles;
    // Store current particles vector into temporary vector
    // assign method erases a vector and copies the specified elements to the empty vector.
    temp_particles.assign(particles.begin(), particles.end());

    //std::cout << "particles size before assigning: " << particles.size() << endl;
    particles.clear();

    for (size_t i = 0; i < num_particles; i++)
    {
        int particle_index = distribution(generator);
        particles.push_back(temp_particles[particle_index]);
    }

    //std::cout << "particles size after resampling: " << particles.size() << endl;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
    const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    // Clear out any associations 
    particle.associations.clear();
    particle.sense_x.clear();
    particle.sense_y.clear();

    // Set current associations
    particle.associations = associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
    vector<int> v = best.associations;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
    vector<double> v = best.sense_x;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
    vector<double> v = best.sense_y;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}
