package de.dikodam.utils

import java.time.Instant
import kotlin.math.E
import kotlin.math.pow
import kotlin.random.Random

// v -> v' = v - n * VC
// where v is the current position (weights and biases? v = x * w + b ? )
// v' is the new position
// n is the rate of learning (rate of change)
// VC is the change in cost

val rng by lazy {
    RNG()
}

class RNG {
    private val random = Random(Instant.now().toEpochMilli())

    fun nextDouble(): Double {
        return random.nextDouble(-1.0, 1.0)
    }

}

