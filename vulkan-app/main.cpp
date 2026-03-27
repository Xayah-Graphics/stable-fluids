#include <cstdio>

import app;
import std;

int main() {
    try {
        app::SmokeApp smoke_app{};
        return smoke_app.run();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
