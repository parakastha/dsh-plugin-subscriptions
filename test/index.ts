/**
 * Test entry so `node --test lib-test/test/` (a directory argument resolves
 * to `lib-test/test/index.js`) picks up every compiled spec.
 */
import './translate.spec.js'
import './models.spec.js'
import './claude-continuity.spec.js'
import './tools.spec.js'
import './rpc.spec.js'
import './usage.spec.js'
