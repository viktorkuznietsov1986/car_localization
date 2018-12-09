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

const double epsilon = 1e-5;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	if (!is_initialized) {

		default_random_engine gen;
		num_particles = 10; // will sort this out later

		normal_distribution<double> dist_x(x, std[0]);
		normal_distribution<double> dist_y(y, std[1]);
		normal_distribution<double> dist_theta(theta, std[2]);

		for (int i = 0; i < num_particles; ++i) {
			double sample_x, sample_y, sample_theta;

			Particle particle;
			particle.id = i;
			particle.x = dist_x(gen);
      particle.y = dist_y(gen);
      particle.theta = dist_theta(gen);
			particle.weight = 1;
			particles.push_back(particle);

			weights.push_back(1);
		}

		is_initialized = true;
	}
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	default_random_engine gen;

	bool use_linear_model = (fabs(yaw_rate) < epsilon);

	auto velocity_to_yaw = use_linear_model ? 1 : (velocity/yaw_rate);

	for (auto& particle : particles) {

		if (use_linear_model) {
			particle.x += velocity*delta_t*cos(particle.theta);
			particle.y += velocity*delta_t*sin(particle.theta);
		}
		else {
			particle.x += velocity_to_yaw*(sin(particle.theta + yaw_rate*delta_t) - sin(particle.theta));
			particle.y += velocity_to_yaw*(cos(particle.theta) - cos(particle.theta + yaw_rate*delta_t));
			particle.theta += yaw_rate*delta_t;
		}

		// add noise
		normal_distribution<double> dist_x(particle.x, std_pos[0]);
		normal_distribution<double> dist_y(particle.y, std_pos[1]);
		normal_distribution<double> dist_theta(particle.theta, std_pos[2]);

		particle.x = dist_x(gen);
		particle.y = dist_y(gen);
		particle.theta = dist_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.
	//for ()
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
	double std_x = std_landmark[0];
	double std_y = std_landmark[1];

	weights.clear();

	double weights_sum = 0;

	auto gauss_norm = 1.0/(2.0*M_PI*std_x*std_y);
	cout << "normalizer = " << gauss_norm << endl;


	for (int i = 0; i < particles.size(); ++i) {
		auto particle = particles[i];
		particle.associations.clear();
		for (const auto& observation: observations) {

			float xm = particle.x + cos(particle.theta)*observation.x - sin(particle.theta)*observation.y;
			float ym = particle.y + sin(particle.theta)*observation.x + cos(particle.theta)*observation.y;

			if (dist(particle.x, particle.y, xm, ym) > sensor_range) {
				continue;
			}

			int idm = 0;
			float min_distance = INFINITY;
			// find the closest landmark
			for (const auto& landmark: map_landmarks.landmark_list) {
				float distance = dist(xm, ym, landmark.x_f, landmark.y_f);
				if (distance < min_distance) {
					idm = landmark.id_i;
					min_distance = distance;
				}
			}

			particle.associations.push_back(idm);
		}

		particle.weight = 1.0;

		for (int j = 0; j < particle.associations.size(); ++j) {
			auto observation = observations[j];
			auto predicted = map_landmarks.landmark_list[particle.associations[j]];

			float xm = particle.x + cos(particle.theta)*observation.x - sin(particle.theta)*observation.y;
			float ym = particle.y + sin(particle.theta)*observation.x + cos(particle.theta)*observation.y;

			auto exponent = (pow(xm-predicted.x_f, 2)/(2.0*pow(std_x, 2)) + pow(ym-predicted.y_f, 2)/(2.0*pow(std_y, 2)));
			// ((x_obs - mu_x)**2)/(2 * sig_x**2) + ((y_obs - mu_y)**2)/(2 * sig_y**2

			cout << "exponent = " << exponent << endl;

			double p = gauss_norm*exp(-exponent);

			cout << "p = " << p << endl;

			particle.weight *= p;
		}

		weights.push_back(particle.weight);

		weights_sum += particle.weight;
	}

	cout << "before normalization = " << weights[0] << endl;

	// normalize weights
	for (auto& weight: weights) {
		weight /= weights_sum;
	}

	cout << "after normalization = " << weights[0] << endl;
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	std::random_device rd;
  std::mt19937 gen(rd());

	std::discrete_distribution<> d(begin(weights), end(weights));

	std::vector<Particle> sampled_particles;

	for (int i = 0; i < particles.size(); ++i) {
		sampled_particles.push_back(particles[d(gen)]);
	}
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
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
