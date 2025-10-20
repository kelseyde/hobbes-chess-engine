use crate::board::bitboard::Bitboard;
use crate::board::castling::{CastleSafety, CastleTravel};
use crate::board::file::File;
use crate::board::movegen::is_attacked;
use crate::board::moves::Move;
use crate::board::piece::Piece;
use crate::board::piece::Piece::{Bishop, Queen};
use crate::board::rank::Rank;
use crate::board::side::Side::{Black, White};
use crate::board::square::Square;
use crate::board::{attacks, ray, Board};

impl Board {
    pub fn is_pseudo_legal(&self, mv: &Move) -> bool {
        if !mv.exists() {
            return false;
        }

        let from = mv.from();
        let to = mv.to();

        if from == to {
            // Cannot move to the same square
            return false;
        }

        let pc = self.piece_at(from);
        let us = self.us();
        let them = self.them();
        let occ = us | them;
        let captured = self.captured(mv);

        // Can't move without a piece
        if pc.is_none() {
            return false;
        }

        let pc = pc.unwrap();

        // Cannot move a piece that is not ours
        if !self.us().contains(from) {
            return false;
        }

        // Cannot capture our own piece
        if us.contains(to) {
            return false;
        }

        if let Some(captured) = captured {
            // Cannot capture a king
            if captured == Piece::King {
                return false;
            }
        }

        if mv.is_castle() {
            // Can only castle with the king
            if pc != Piece::King {
                return false;
            }

            let rank = if self.stm == White {
                Rank::One
            } else {
                Rank::Eight
            };
            let rank_bb = rank.to_bb();
            if !rank_bb.contains(from) || !rank_bb.contains(to) {
                // Castling must be on the first rank
                return false;
            }

            let kingside_sq = if self.stm == White {
                Square(6)
            } else {
                Square(62)
            };
            let queenside_sq = if self.stm == White {
                Square(2)
            } else {
                Square(58)
            };

            // Castling must be to the kingside or queenside square
            if to != kingside_sq && to != queenside_sq {
                return false;
            }

            // Cannot castle kingside if no rights
            if to == kingside_sq && !self.has_kingside_rights(self.stm) {
                return false;
            }

            // Cannot castle queenside if no rights
            if to == queenside_sq && !self.has_queenside_rights(self.stm) {
                return false;
            }

            let kingside = to == kingside_sq;
            let travel_sqs = if kingside {
                if self.stm == White {
                    CastleTravel::WKS
                } else {
                    CastleTravel::BKS
                }
            } else if self.stm == White {
                CastleTravel::WQS
            } else {
                CastleTravel::BQS
            };

            // Cannot castle through occupied squares
            if !(occ & travel_sqs).is_empty() {
                return false;
            }

            let safety_sqs = if kingside {
                if self.stm == White {
                    CastleSafety::WKS
                } else {
                    CastleSafety::BKS
                }
            } else if self.stm == White {
                CastleSafety::WQS
            } else {
                CastleSafety::BQS
            };

            // Cannot castle through check
            if is_attacked(safety_sqs, self.stm, occ, self) {
                return false;
            }
        }

        if pc == Piece::Pawn {
            if mv.is_ep() {
                // Cannot en passant if no en passant square
                if self.ep_sq.is_none() {
                    return false;
                }

                let ep_capture_sq = self.ep_capture_sq(to);

                // Cannot en passant if no pawn on the capture square
                if !them.contains(ep_capture_sq) {
                    return false;
                }
            }

            let from_rank = Rank::of(from);
            let to_rank = Rank::of(to);

            // Cannot move a pawn backwards
            if (self.stm == White && to_rank < from_rank)
                || (self.stm == Black && to_rank > from_rank)
            {
                return false;
            }

            let promo_rank = if self.stm == White {
                Rank::Eight
            } else {
                Rank::One
            };

            // Cannot promote a pawn if not to the promotion rank
            if mv.is_promo() && !promo_rank.to_bb().contains(to) {
                return false;
            }

            let from_file = File::of(from);
            let to_file = File::of(to);

            // Pawn captures
            if from_file != to_file {
                // Must be capturing a piece
                if captured.is_none() {
                    return false;
                }

                // Must capture on an adjacent file
                if to_file as usize != from_file as usize + 1
                    && to_file as usize != from_file as usize - 1
                {
                    return false;
                }

                if mv.is_ep() {
                    if self.ep_sq.is_none() {
                        // Cannot en passant if no en passant square
                        return false;
                    }
                    if self.piece_at(self.ep_sq.unwrap()) != Some(Piece::Pawn) {
                        // Cannot en passant if no pawn on the en passant square
                        return false;
                    }
                } else {
                    // Must be a valid capture square
                    if !attacks::pawn(from, self.stm).contains(to) {
                        return false;
                    }
                }

                true
            } else {
                // Cannot capture a piece with a pawn push
                if captured.is_some() {
                    return false;
                }

                if mv.is_double_push() {
                    let start_rank = if self.stm == White {
                        Rank::Two
                    } else {
                        Rank::Seven
                    };
                    // Cannot double push a pawn if not on the starting rank
                    if !start_rank.to_bb().contains(from) {
                        return false;
                    }

                    let between_sq = if self.stm == White {
                        Square(from.0 + 8)
                    } else {
                        Square(from.0 - 8)
                    };
                    // Cannot double push a pawn if the square in between is occupied
                    if occ.contains(between_sq) {
                        return false;
                    }

                    // Cannot double push to an occupied square
                    !occ.contains(to)
                } else {
                    // Must be a single push
                    if to.0
                        != if self.stm == White {
                            from.0 + 8
                        } else {
                            from.0 - 8
                        }
                    {
                        return false;
                    }

                    !occ.contains(to)
                }
            }
        } else {
            // Can't make a pawn-specific move with a non-pawn
            if mv.is_ep() || mv.is_promo() || mv.is_double_push() {
                return false;
            }

            let attacks = attacks::attacks(from, pc, self.stm, occ);
            attacks.contains(to)
        }
    }

    /// This function assumes that the move is pseudo-legal
    pub fn is_legal(&self, mv: &Move) -> bool {
        let from = mv.from();
        let to = mv.to();

        let king_sq = self.king_sq(self.stm);
        let threats = self.threats;
        let pinned = self.pinned[self.stm];

        if mv.is_ep() {
            let from_bb = Bitboard::of_sq(from);
            let to_bb = Bitboard::of_sq(to);
            let ep_bb = Bitboard::of_sq(self.ep_capture_sq(to));
            let occ = self.occ() ^ from_bb ^ to_bb ^ ep_bb;

            let diagonals = self.their(Bishop) | self.their(Queen);
            let orthogonals = self.their(Piece::Rook) | self.their(Queen);

            let diagonal_attacks = attacks::bishop(king_sq, occ) & diagonals;
            let orthogonal_attacks = attacks::rook(king_sq, occ) & orthogonals;

            return (diagonal_attacks | orthogonal_attacks).is_empty();
        }

        if mv.is_castle() {
            return !threats.contains(to) && !(self.frc && pinned.contains(to));
        }

        if let Some(Piece::King) = self.piece_at(from) {
            return !threats.contains(to);
        }

        if pinned.contains(from) {
            let moving_along_pin_ray = ray::between(king_sq, from).contains(to)
                || ray::between(king_sq, to).contains(from);
            return self.checkers.is_empty() && moving_along_pin_ray;
        }

        if self.checkers.count() > 1 {
            return false;
        }

        if self.checkers.is_empty() {
            return true;
        }

        let checker = self.checkers.lsb();
        (self.checkers | ray::between(king_sq, checker)).contains(to)
    }
}
