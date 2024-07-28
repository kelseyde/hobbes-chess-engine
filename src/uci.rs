use std::io;

pub fn run() {

    loop {
        let mut command = String::new();
        io::stdin()
            .read_line(&mut command)
            .expect("info error failed to parse command");

        match command.split_ascii_whitespace().next().unwrap() {
            "uci" =>          handle_uci(),
            "isready" =>      handle_isready(),
            "setoption" =>    handle_setoption(command),
            "ucinewgame" =>   handle_ucinewgame(),
            "position" =>     handle_position(command),
            "go" =>           handle_go(command),
            "stop" =>         handle_stop(),
            "ponderhit" =>    handle_ponderhit(),
            "eval" =>         handle_eval(),
            "datagen" =>      handle_datagen(command),
            "help" =>         handle_help(),
            "quit" =>         handle_quit(),
            _ => {}
        }
    }

}

fn handle_uci() {
    println!("id name Hobbes");
    println!("id author Dan Kelsey");
    println!("uciok");
}

fn handle_isready() {
    println!("readyok");
}

fn handle_setoption(command: String) {

}

fn handle_ucinewgame() {

}

fn handle_position(command: String) {

}

fn handle_go(command: String) {

}

fn handle_ponderhit() {

}

fn handle_eval() {

}

fn handle_datagen(command: String) {

}

fn handle_stop() {

}

fn handle_help() {
    println!("the following commands are available:");
    println!("uci         -- print engine info");
    println!("isready     -- check if engine is ready");
    println!("setoption   -- set engine options");
    println!("ucinewgame  -- clear the board and set up a new game");
    println!("position    -- set up the board position");
    println!("go          -- start searching for the best move");
    println!("stop        -- stop searching and return the best move");
    println!("ponderhit   -- opponent played the expected move");
    println!("eval        -- evaluate the current position");
    println!("datagen     -- generate training data for neural network");
    println!("quit        -- exit the application");
}

fn handle_quit() {
    std::process::exit(0);
}