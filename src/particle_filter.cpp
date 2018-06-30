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
    // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
    //   x, y, theta and their uncertainties from GPS) and all weights to 1.
    // Add random Gaussian noise to each particle.
    // NOTE: Consult particle_filter.h for more information about this method (and others in this file).

    // Number of particles
    num_particles = 100;

    std::default_random_engine generate;

    // Normal distribution of sensor noises
    std::normal_distribution<double> std_x(x, std[0]);
    std::normal_distribution<double> std_y(y, std[1]);
    std::normal_distribution<double> std_theta(theta, std[2]);

    // Initialize the particles
    for(size_t i=0; i < num_particles; i++)
    {
        Particle P;
        P.id            = i; //int id
        P.x             = std_x(generate); //double x
        P.y             = std_y(generate); //double y
        P.theta         = std_theta(generate); //double theta
        P.weight        = 1.0; //double weight

        particles.push_back(P);
        weights.push_back(1.0);
    }

    // Set the is_initialized flag to true
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    // TODO: Add measurements to each particle and add random Gaussian noise.
    // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
    //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
    //  http://www.cplusplus.com/reference/random/default_random_engine/

    double prediction_x;
    double prediction_y;
    double prediction_theta;
    std::default_random_engine generate;


    for(size_t i=0; i<num_particles; i++)
    {
        // calculate the new states

        if(fabs(yaw_rate) < 0.0001)//Check not divisible by zero
        {
            prediction_x        = particles[i].x + velocity * delta_t * cos(particles[i].theta);
            prediction_y        = particles[i].y + velocity * delta_t * sin(particles[i].theta);
            prediction_theta    = particles[i].theta;
        }
        else
        {
            prediction_x        = particles[i].x + velocity/yaw_rate * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
            prediction_y        = particles[i].y + velocity/yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t)) ;
            prediction_theta    = particles[i].theta + yaw_rate * delta_t;
        }

        std::normal_distribution<double> std_x(prediction_x, std_pos[0]);
        std::normal_distribution<double> std_y(prediction_y, std_pos[1]);
        std::normal_distribution<double> std_theta(prediction_theta, std_pos[2]);

        particles[i].x      = std_x(generate);
        particles[i].y      = std_y(generate);
        particles[i].theta  = std_theta(generate);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
    // TODO: Find the predicted measurement that is closest to each observed measurement and assign the
    //   observed measurement to this particular landmark.
    // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
    //   implement this method and use it as a helper during the updateWeights phase.

    for (size_t i= 0; i<observations.size(); i++)
    {
        double observation_x = observations[i].x;
        double observation_y = observations[i].y;
        double minimum_distance = std::numeric_limits<double>::max();
        double id = -1;//init with a placeholder

        for (size_t j=0; j<predicted.size(); j++)
        {
            double x = predicted[j].x;
            double y = predicted[j].y;

            double farness = dist(observation_x, observation_y, x, y);

            //Nearest landmart
            if(farness < minimum_distance)
            {
                minimum_distance = farness;
                id = predicted[j].id;
            }
        }
        observations[i].id = id;//Nearest landmark ID
    }
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

    double gaussian_normalizer = 1 / (2 * M_PI * std_landmark[0] * std_landmark[1]);

    for (size_t i = 0; i < num_particles; i++)
    {
        double particle_x = particles[i].x;
        double particle_y = particles[i].y;
        double particle_theta = particles[i].theta;

        std::vector<LandmarkObs> filtered_landmarks;
        std::vector<LandmarkObs> transformedobjects;

        // in map coordinate system
        for (size_t j = 0; j < observations.size(); j++)
        {
            LandmarkObs transformedobject;
            transformedobject.x = particle_x + (cos(particle_theta) * observations[j].x) - (sin(particle_theta) * observations[j].y);
            transformedobject.y = particle_y + (sin(particle_theta) * observations[j].x) + (cos(particle_theta) * observations[j].y);
            transformedobject.id = observations[j].id;
            transformedobjects.push_back(transformedobject);
        }

        //Filtering
        for (size_t j = 0; j < map_landmarks.landmark_list.size(); j++)
        {
            double distance = dist(particle_x, particle_y, map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f);
            if ( distance <= sensor_range)
            {
                filtered_landmarks.push_back(LandmarkObs {map_landmarks.landmark_list[j].id_i, map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f});
            }
        }

        dataAssociation(filtered_landmarks, transformedobjects);

        particles[i].weight = 1.0;

        // Calculating new weights
        for (size_t j = 0; j < transformedobjects.size(); j++)
        {

            double predicted_landmark_x, predicted_landmark_y;

            for (size_t k = 0; k < filtered_landmarks.size(); k++)
            {
                if (filtered_landmarks[k].id == transformedobjects[j].id)
                {
                    predicted_landmark_x = filtered_landmarks[k].x;
                    predicted_landmark_y = filtered_landmarks[k].y;

                    double multi_gaussian = gaussian_normalizer * exp( -1.0 * ( pow(predicted_landmark_x - transformedobjects[j].x, 2) / (2 * pow(std_landmark[0], 2))
                                                                            + ( pow(predicted_landmark_y - transformedobjects[j].y, 2) / (2* pow (std_landmark[1], 2))) ) );

                    particles[i].weight *= multi_gaussian;
                }
            }
        }
        weights[i] = particles[i].weight;
    }
}

void ParticleFilter::resample() {
    // TODO: Resample particles with replacement with probability proportional to their weight.
    // NOTE: You may find std::discrete_distribution helpful here.
    //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

    std::vector<Particle> resample_particles;
    std::default_random_engine generate;
    std::discrete_distribution<int> distribution(weights.begin(), weights.end());

    for(size_t i = 0; i < num_particles; i++)
    {
        resample_particles.push_back(particles[distribution(generate)]);
    }
    particles = resample_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                         const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

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
