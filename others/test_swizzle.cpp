#include <iostream>
#include <vector>

std::vector<unsigned> rotate_index(unsigned int offset) {
	unsigned int mask15 = 1 << 15;
	unsigned int offset15 = (offset & mask15) >> 15;
	std::vector<unsigned> result;
	if (offset15) {
		unsigned offset1_0 = (offset & 0x3);
		unsigned offset3_2 = (offset & 0xc) >> 2;
		unsigned offset5_4 = (offset & 0x30) >> 4;
		unsigned offset7_6 = (offset & 0xc0) >> 6;

		for (unsigned i = 0; i < 64; i += 4) {
			unsigned i0 = i + offset1_0;
			unsigned i1 = i + offset3_2;
			unsigned i2 = i + offset5_4;
			unsigned i3 = i + offset7_6;
			result.push_back(i0);
			result.push_back(i1);
			result.push_back(i2);
			result.push_back(i3);
		}
	}
	else {
		unsigned mask_xor = (offset & (0x1f << 10)) >> 10;
		unsigned mask_or = (offset & (0x1f << 5)) >> 5;
		unsigned mask_and = (offset & 0x1f);
		for (unsigned i = 0; i < 64; ++i) {
			int j = (((i & 0x1f) & mask_and) | mask_or) ^ mask_xor;
			j = j | (i & 0x20);
			result.push_back(j);
		}
	}

	return result;
}

int main(int argc, char **argv) {
	if (argc != 2) {
		std::cout << "Usage: " << argv[0] << " offset" << std::endl;
		return 1;
	}

	unsigned offset = std::atoi(argv[1]);
	auto result = rotate_index(offset);

	for (std::size_t i = 0; i < result.size(); ++i) {
		std::cout << i << "\t" << result[i] << std::endl;
	}

	return 0;
}

