/** Reading the joint strip. */
import { isOneSidedJoint, type JointUnit } from './jointScale'

/**
 * Reading the joint strip — UX_REVIEW #4, the part still missing. The numbers and the
 * sparklines landed already.
 */

/** `shoulder_pan.pos` → `Shoulder pan`. Never translates, only reformats. */
export function humanJointName(key: string): string {
  const bare = key.replace(/(_pos|\.pos)$/, '')
  const words = bare.replace(/[_.]+/g, ' ').replace(/\s+/g, ' ').trim()
  if (!words) return key
  return words.charAt(0).toUpperCase() + words.slice(1)
}

/** Labels for a whole strip. */
export function humanJointNames(keys: readonly string[]): string[] {
  const human = keys.map(humanJointName)
  const collides = new Set(human).size !== human.length
  return collides ? keys.slice() : human
}

/**
 * One line under the strip: what the numbers are in, and what the two graphics mean.
 * `jointNames` is used only to say whether a gripper is among them, since that is the row
 * whose unit differs.
 */
export function stripLegend(
  unit: JointUnit,
  windowMs: number,
  jointNames: readonly string[] = [],
): string {
  const secs = Math.round(windowMs / 1000)
  // ONE definition of "this joint is on its own 0…100 scale": jointScale's, the module that
  // actually puts it there.
  const hasGripper = jointNames.some(isOneSidedJoint)
  const units = unit === 'radian'
    ? 'values in radians'
    : hasGripper
      ? 'values in degrees, gripper on its own 0…100 scale'
      : 'values in degrees'
  return `${units} · bar = position within travel seen so far · line = last ${secs}s`
}
