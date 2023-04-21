use quarkstrom;

fn main() {
    pollster::block_on(quarkstrom::run());
}
