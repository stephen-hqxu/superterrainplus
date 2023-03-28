//Catch2
#include <catch2/catch_test_macros.hpp>
//Matcher
#include <catch2/matchers/catch_matchers_floating_point.hpp>

//Linear Algebra
#include <SuperTerrain+/Utility/Algebra/STPMatrix4x4d.h>
#include <SuperTerrain+/Utility/Algebra/STPVector4d.h>

//GLM
#include <glm/geometric.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_inverse.hpp>

#include <type_traits>

using SuperTerrainPlus::STPVector4d, SuperTerrainPlus::STPMatrix4x4d;

using glm::dmat3;
using glm::vec4;
using glm::dvec4;
using glm::mat4;
using glm::dmat4;
using glm::inverse;
using glm::transpose;
using glm::dot;

#define COMP_FLOATING_POINT(X, TYPE) Catch::Matchers::WithinULP(X, 3ull)
#define COMP_FLOAT(X) COMP_FLOATING_POINT(X, float)
#define COMP_DOUBLE(X) COMP_FLOATING_POINT(X, double)

//This compare function consider tolerance of floating point comparison
template<class T>
static void compareVector(const T& val, const T& expected) {
	for (int row = 0; row < 4; row++) {
		if constexpr (std::is_same_v<T, dvec4>) {
			REQUIRE_THAT(val[row], COMP_DOUBLE(expected[row]));
		} else {
			REQUIRE_THAT(val[row], COMP_FLOAT(expected[row]));
		}
	}
}

template<class T>
static void compareMatrix(const T& val, const T& expected) {
	for (int col = 0; col < 4; col++) {
		compareVector(val[col], expected[col]);
	}
}

SCENARIO("STPMatrix4x4d can be used to perform matrix operation", "[Algebra][STPMatrix4x4d]") {
	
	WHEN("The matrix is default initialised") {
		const STPMatrix4x4d Mat;

		THEN("All components are initialised to zero and data can be retrieved") {
			constexpr static dmat4 Zero = dmat4(0.0);
			const dmat4 MatData = static_cast<dmat4>(Mat);

			REQUIRE(MatData == Zero);
		}

	}

	GIVEN("A regular piece of matrix data") {
		//column-major
		constexpr static dmat4 Data = dmat4(
			dvec4(0.0, 4.0, 8.0, 12.0),
			dvec4(1.0, -5.5, 9.0, 13.0),
			dvec4(2.0, 6.0, -10.5, 14.0),
			dvec4(3.0, 7.0, 11.0, 15.0)
		);
		const STPMatrix4x4d Mat(Data);

		THEN("Matrix can be loaded") {

			AND_THEN("Data can be unloaded and read back") {
				const mat4 MatfData = static_cast<mat4>(Mat);
				const dmat4 MatdData = static_cast<dmat4>(Mat);

				compareMatrix(MatfData, static_cast<mat4>(Data));
				compareMatrix(MatdData, Data);
			}

		}

		WHEN("Matrix is transposed") {
			const auto MatT = Mat.transpose();

			THEN("Result of transpose is verified") {
				const dmat4 MatTData = static_cast<dmat4>(MatT),
					ActualMatT = transpose(Data);

				compareMatrix(MatTData, ActualMatT);
			}

		}

		WHEN("Matrix is inverted") {
			const auto MatInv = Mat.inverse();

			THEN("Result of inverse is verified") {
				const dmat4 MatInvData = static_cast<dmat4>(MatInv),
					ActualMatInv = inverse(Data);

				compareMatrix(MatInvData, ActualMatInv);
			}
		}

		WHEN("Matrix multiplication is performed") {

			AND_WHEN("Matrix is multiplied with another matrix") {
				const auto MatSq = Mat * Mat;

				THEN("Result of matrix-matrix multiplication is verified") {
					const dmat4 MatSqData = static_cast<dmat4>(MatSq),
						ActualMatSq = Data * Data;

					compareMatrix(MatSqData, ActualMatSq);
				}

			}

			AND_WHEN("Matrix is multiplied with a vector") {
				constexpr static dvec4 VecRaw = dvec4(15.5, -16.5, -17.5, 18.5);
				const STPVector4d Vec(VecRaw);
				const auto MatVec = Mat * Vec;

				THEN("Result of matrix-vector multiplication is verified") {
					const dvec4 MatVecData = static_cast<dvec4>(MatVec),
						ActualMatVec = Data * VecRaw;

					compareVector(MatVecData, ActualMatVec);
				}

			}

		}

		WHEN("Matrix is cast to a 3-by-3 matrix") {
			const auto Mat3 = Mat.asMatrix3x3d();

			THEN("The new matrix is a 4-by-4 matrix correctly representing the 3-by-3 matrix") {
				const dmat4 Mat3Data = static_cast<dmat4>(Mat3),
					ActualMat3 = dmat4(dmat3(Data));

				compareMatrix(Mat3Data, ActualMat3);
			}

		}

	}

}

SCENARIO("STPVector4d can be used to perform vector operation", "[Algebra][STPVector4d]") {

	WHEN("The vector is default initialised") {
		const STPVector4d Vec;

		THEN("It is initialised to zero and data can be retrieved") {
			constexpr static dvec4 Zero = dvec4(0.0);
			const dvec4 VecData = static_cast<dvec4>(Vec);

			REQUIRE(VecData == Zero);
		}

	}

	GIVEN("A regular piece of vector data") {
		constexpr static dvec4 Data = dvec4(1.5, -2.5, 3.5, 4.5),
			AnotherData = dvec4(-5.5, 6.5, -7.5, 8.5);
		const STPVector4d Vec(Data);

		THEN("Data can be loaded into the vector") {

			AND_THEN("Data can be unloaded and read back") {
				const vec4 VecfData = static_cast<vec4>(Vec);
				const dvec4 VecdData = static_cast<dvec4>(Vec);

				compareVector(VecfData, static_cast<vec4>(Data));
				compareVector(VecdData, Data);
			}

		}

		WHEN("Vector arithmetic is performed") {
			const STPVector4d RHSVec(AnotherData);

			THEN("All arithmetic operations are performed correctly") {
				const auto Plus = Vec + RHSVec,
					Div = Vec / RHSVec;
				const dvec4 PlusData = static_cast<dvec4>(Plus),
					DivData = static_cast<dvec4>(Div);

				compareVector(PlusData, Data + AnotherData);
				compareVector(DivData, Data / AnotherData);
			}

		}

		WHEN("Vector dot product is performed") {
			const STPVector4d AnotherVec(AnotherData);
			const double VecDot = Vec.dot(AnotherVec);

			THEN("Result of dot product is verified with reasonable precision") {
				const double ActualVecDot = dot(Data, AnotherData);

				REQUIRE_THAT(VecDot, COMP_DOUBLE(ActualVecDot));
			}

		}

		WHEN("Vector broadcast is performed") {
			const auto BroadcastVec = Vec.broadcast<STPVector4d::STPElement::Y>();

			THEN("The broadcast vector should contain the element of the original vector") {
				const dvec4 BroadcastData = static_cast<dvec4>(BroadcastVec);
				constexpr static dvec4 ActualBroadcast = dvec4(Data.y);

				compareVector(BroadcastData, ActualBroadcast);
			}

		}

	}

}