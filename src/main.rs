use crate::uci::UCI;

pub mod attacks;
pub mod bench;
pub mod board;
pub mod fen;
pub mod magics;
pub mod movegen;
pub mod moves;
pub mod perft;
pub mod search;
pub mod thread;
pub mod uci;
pub mod zobrist;
pub mod tt;
pub mod history;
pub mod see;
pub mod types;
pub mod network;
mod time;
mod movepicker;

fn main() {

    // let startpos = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    // let mut board = Board::from_fen(startpos);
    // let mut nnue = Box::new(NNUE::default());
    // nnue.activate(&board);
    // println!("startpos eval: {}", nnue.evaluate(&board));
    //
    // let lostpos = "rnbqkbnr/pppppppp/8/8/8/8/8/3QK3 w kq - 0 1";
    // let mut lost_board = Board::from_fen(lostpos);
    // let mut lost_nnue = Box::new(NNUE::default());
    // lost_nnue.activate(&lost_board);
    // println!("lostpos eval: {}", lost_nnue.evaluate(&lost_board));
    //
    // let wonpos = "rn2k1nr/ppp2ppp/8/4P3/2P3b1/8/PP1B1KPP/RN1q1BR1 b kq - 1 10";
    // let mut won_board = Board::from_fen(wonpos);
    // let mut won_nnue = Box::new(NNUE::default());
    // won_nnue.activate(&won_board);
    // println!("wonpos eval: {}", won_nnue.evaluate(&won_board));


    // let mut board = Board::from_fen(fen::STARTPOS);
    // let mut nnue_1 = Box::new(NNUE::default());
    // nnue_1.activate(&board);
    // let mv = Move::parse_uci("e2e4");
    // let pc = Pawn;
    // board.make(&mv);
    // nnue_1.update(&mv, pc, None, &board);
    //
    // let eval_1 = nnue_1.evaluate(&board);
    //
    // let mut nnue_2 = Box::new(NNUE::default());
    // nnue_2.activate(&board);
    //
    // let eval_2 = nnue_2.evaluate(&board);
    //
    // assert_eq!(eval_1, eval_2);

    let args: Vec<String> = std::env::args().collect();
    UCI::new().run(&args);
}


